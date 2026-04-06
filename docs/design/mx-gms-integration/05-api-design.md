# 5. API Design

[< Back to Overview](README.md)

## 1. Weight Loader Protocol

```python
# tensorrt_llm/_torch/weight_loaders/base.py

from typing import Protocol, Dict, Optional, Tuple
from dataclasses import dataclass
import torch

@dataclass
class TensorDescriptor:
    """Describes a GPU tensor for P2P registration."""
    name: str
    data_ptr: int
    size_bytes: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    is_contiguous: bool
    storage_offset: int = 0
    # View reconstruction metadata (for non-contiguous tensors)
    storage_name: Optional[str] = None
    view_shape: Optional[Tuple[int, ...]] = None
    view_stride: Optional[Tuple[int, ...]] = None

class WeightLoaderProtocol(Protocol):
    """Interface for custom weight loading backends."""

    def load_weights(
        self,
        model: torch.nn.Module,
        mapping: "Mapping",
        config: "PretrainedConfig",
    ) -> None:
        """Load weights into model from custom source."""
        ...

    def get_tensor_descriptors(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, TensorDescriptor]:
        """Return tensor metadata for P2P registration."""
        ...

    def supports_source_mode(self) -> bool:
        """Whether this loader can act as P2P source."""
        ...

    def cleanup(self) -> None:
        """Release any resources held by the loader."""
        ...
```

## 2. Tensor Enumeration API

```python
# tensorrt_llm/_torch/utils/tensor_utils.py

def enumerate_model_tensors(
    model: torch.nn.Module,
    include_buffers: bool = True,
    include_quantization_scales: bool = True,
    deduplicate_by_storage: bool = True,
) -> Dict[str, TensorDescriptor]:
    """
    Enumerate all GPU tensors in a model for P2P registration.

    Handles:
        - Parameters and buffers
        - Tied weights (deduplicated by data_ptr)
        - Non-contiguous views (reports underlying storage + view metadata)
        - Quantization scales (weight_scale_inv, etc.)
        - Aliased layers from post_load_weights()

    Returns:
        Dictionary mapping tensor names to descriptors
    """
    descriptors = {}
    seen_data_ptrs = {}  # data_ptr -> canonical name

    for name, param in model.named_parameters():
        ptr = param.data.data_ptr()

        if deduplicate_by_storage and ptr in seen_data_ptrs:
            # Tied weight — record alias, skip duplicate registration
            descriptors[name] = TensorDescriptor(
                name=name,
                data_ptr=ptr,
                size_bytes=0,  # Alias — no additional memory
                dtype=param.dtype,
                shape=param.shape,
                is_contiguous=param.is_contiguous(),
                storage_name=seen_data_ptrs[ptr],
            )
            continue

        seen_data_ptrs[ptr] = name

        if param.is_contiguous():
            descriptors[name] = TensorDescriptor(
                name=name,
                data_ptr=ptr,
                size_bytes=param.nelement() * param.element_size(),
                dtype=param.dtype,
                shape=param.shape,
                is_contiguous=True,
            )
        else:
            # Non-contiguous: register underlying storage
            storage = param.untyped_storage()
            descriptors[name] = TensorDescriptor(
                name=name,
                data_ptr=storage.data_ptr(),
                size_bytes=storage.nbytes(),
                dtype=param.dtype,
                shape=param.shape,
                is_contiguous=False,
                view_shape=param.shape,
                view_stride=param.stride(),
            )

    if include_buffers:
        for name, buf in model.named_buffers():
            if buf.device.type != 'cuda':
                continue
            ptr = buf.data.data_ptr()
            if deduplicate_by_storage and ptr in seen_data_ptrs:
                continue
            seen_data_ptrs[ptr] = name
            descriptors[name] = TensorDescriptor(
                name=name,
                data_ptr=ptr,
                size_bytes=buf.nelement() * buf.element_size(),
                dtype=buf.dtype,
                shape=buf.shape,
                is_contiguous=buf.is_contiguous(),
            )

    return descriptors
```

## 3. Memory Allocator Hook

```python
# Addition to tensorrt_llm/_torch/pyexecutor/model_loader.py

from typing import Callable, Optional

AllocatorFn = Callable[[int, torch.dtype, str], torch.Tensor]

class ModelLoader:
    def __init__(
        self,
        llm_args: "TorchLlmArgs",
        mapping: "Mapping",
        *,
        custom_allocator: Optional[AllocatorFn] = None,
        post_load_callback: Optional[Callable[[torch.nn.Module], None]] = None,
        **kwargs,
    ):
        """
        Args:
            custom_allocator: Function(size_bytes, dtype, tag) -> torch.Tensor
                             Routes GPU allocation through custom backend (e.g., GMS)
            post_load_callback: Called after model loaded and post_load_weights()
                               completed. Use for P2P tensor registration.
        """
        self._custom_allocator = custom_allocator
        self._post_load_callback = post_load_callback
```

## 4. External Memory Import/Export

```python
# tensorrt_llm/_torch/memory/external_memory.py

def import_cuda_memory(
    fd: int,
    size: int,
    device: int,
) -> torch.Tensor:
    """
    Import external CUDA memory via file descriptor (cuMemImportFromShareableHandle).
    Used by GMS to import memory allocated by the GMS server.
    """
    ...

def export_cuda_memory(
    tensor: torch.Tensor,
) -> Tuple[int, int]:
    """
    Export CUDA memory as file descriptor (cuMemExportToShareableHandle).
    Used by GMS to share memory with other processes.
    """
    ...
```

## 5. GMS Allocator

```python
# tensorrt_llm/_torch/memory/gms_allocator.py

class GMSAllocator:
    """Routes PyTorch GPU allocations through GMS for out-of-process memory management."""

    def __init__(self, gms_client, tag: str = "model_weights"):
        self.gms_client = gms_client
        self.tag = tag
        self._allocations = {}  # ptr -> (size, gms_handle)

    def malloc(self, size: int, device: int, stream) -> int:
        handle = self.gms_client.create_mapping(size=size, tag=self.tag)
        ptr = handle.data_ptr
        self._allocations[ptr] = (size, handle)
        return ptr

    def free(self, ptr: int, size: int, device: int, stream):
        if ptr in self._allocations:
            _, handle = self._allocations.pop(ptr)
            # Only destroy if we're the RW owner
            if self.mode == "rw":
                self.gms_client.destroy_mapping(handle)
            # In RO mode, just release our reference

    def commit(self):
        """Commit all allocations to GMS for sharing."""
        self.gms_client.commit(tag=self.tag)

    def as_torch_allocator(self):
        """Return a torch.cuda.memory.CUDAPluggableAllocator."""
        return torch.cuda.memory.CUDAPluggableAllocator(
            self.malloc, self.free
        )
```

## 6. MX Source Identity

```python
# tensorrt_llm/_torch/weight_loaders/mx_loader.py

@dataclass
class MXSourceIdentity:
    """Content-addressed identity for MX source matching."""
    model_name: str
    dtype: str
    quantization: Optional[str]
    tp_size: int
    pp_size: int
    ep_size: int
    worker_rank: int
    pp_rank: int
    ep_rank: int
    quant_config_hash: Optional[str]  # SHA256 of serialized quant config

    def to_mx_identity(self) -> "mx_proto.SourceIdentity":
        """Convert to MX protobuf SourceIdentity."""
        return mx_proto.SourceIdentity(
            model_name=self.model_name,
            dtype=self.dtype,
            extra_params={
                "quantization": self.quantization or "",
                "tp_size": str(self.tp_size),
                "pp_size": str(self.pp_size),
                "ep_size": str(self.ep_size),
                "worker_rank": str(self.worker_rank),
                "quant_config_hash": self.quant_config_hash or "",
            },
        )
```

## 7. Configuration Schema

```python
# Additions to tensorrt_llm/llmapi/llm_args.py

class TorchLlmArgs(BaseLlmArgs):
    # ... existing fields ...

    # Weight loading format
    load_format: Literal["auto", "hf", "dummy", "mx", "gms", "mx-gms"] = "auto"

    # MX-specific configuration
    mx_server_url: Optional[str] = None
    mx_metadata_backend: Optional[Literal["redis", "kubernetes"]] = None
    mx_heartbeat_interval_secs: int = 30

    # GMS-specific configuration
    gms_socket_path: Optional[str] = None  # Default: /tmp/gms-{device_id}.sock
    gms_mode: Literal["auto", "rw", "ro"] = "auto"
    gms_tag: str = "model_weights"

    # Shorthand
    enable_weight_sharing: bool = False  # Equivalent to load_format="mx-gms"
```

**CLI usage:**
```bash
# MX only (P2P across nodes)
trtllm-serve meta-llama/Llama-3.1-70B --load-format mx --mx-server-url http://mx:8001

# GMS only (sharing within node)
trtllm-serve meta-llama/Llama-3.1-70B --load-format gms --gms-socket-path /tmp/gms-0.sock

# Combined
trtllm-serve meta-llama/Llama-3.1-70B --load-format mx-gms \
    --mx-server-url http://mx:8001 --gms-socket-path /tmp/gms-0.sock

# Shorthand
trtllm-serve meta-llama/Llama-3.1-70B --enable-weight-sharing
```

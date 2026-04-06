# 10. Startup Performance Profiling

[< Back to Overview](README.md)

## E2E Startup Timeline

```
Process Start
    │
    ▼
┌─────────────────┐
│ 1. Python Init  │ ~1-3s
│    & Imports    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Config Load  │ ~0.5-2s
│    & Validate   │
└────────┬────────┘
         ▼
┌─────────────────┐  ◀── CRITICAL BOTTLENECK (Network-bound)
│ 3. Model        │ ~30s - 30min
│    Download     │ MX eliminates this for replicas
└────────┬────────┘
         ▼
┌─────────────────┐  ◀── CRITICAL BOTTLENECK (I/O-bound)
│ 4. Weight       │ ~10s - 5min
│    Loading      │ GMS eliminates this for co-located workers
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. Weight       │ ~5-30s
│    Processing   │ (dtype conversion, quant, TP sharding)
└────────┬────────┘
         ▼
┌─────────────────┐  ◀── SIGNIFICANT DELAY (Compute-bound)
│ 6. Compilation  │ ~10s - 2min
│                 │ (torch.compile, CUDA graphs)
└────────┬────────┘
         ▼
┌─────────────────┐
│ 7. KV Cache     │ ~1-10s
│    Allocation   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 8. Executor +   │ ~1-5s
│    Server Init  │
└────────┬────────┘
         ▼
Ready to Serve
```

## Bottleneck Analysis

| Phase | Duration | Bottleneck | MX/GMS Impact |
|:------|:---------|:----------|:-------------|
| **Model Download** | 30s - 30min | Network I/O | **MX: P2P from existing replica (eliminated)** |
| **Weight Loading** | 10s - 5min | Disk I/O | **GMS: Zero-copy import (~100ms)** |
| **Compilation** | 10s - 2min | Compute | Future: compile cache sharing via MX |
| **Weight Processing** | 5-30s | CPU/GPU | Both sides run identical processing |
| **KV Cache Alloc** | 1-10s | GPU memory | Separate concern (KVBM) |

## Impact Modeling

### Without MX/GMS (Current)

```
N replicas × (download + load + process + compile + init)
= N × (180s + 120s + 15s + 60s + 5s) = N × 380s
```

### With MX Only (Phase 1)

```
1 seed × (download + load + process + compile + init) + (N-1) × (P2P + process + compile + init)
= 380s + (N-1) × (15s + 15s + 60s + 5s) = 380s + (N-1) × 95s
```

### With MX + GMS (Phase 3)

```
1 seed × (download + load + process + compile + init + GMS commit) + (N-1) × (GMS import + compile + init)
= 385s + (N-1) × (0.1s + 60s + 5s) = 385s + (N-1) × 65s
```

### With MX + GMS + Compile Cache (Future)

```
1 seed × full startup + (N-1) × (GMS import + compile cache load + init)
= 385s + (N-1) × (0.1s + 5s + 5s) = 385s + (N-1) × 10s
```

## Proposed Profiling Framework

### StartupProfiler

```python
# tensorrt_llm/_torch/utils/startup_profiler.py

class StartupProfiler:
    """Hierarchical profiler for TRT-LLM startup phases."""

    _instance = None

    def __init__(self):
        self.records = {}
        self.stack = []
        self.enabled = os.environ.get("TRTLLM_PROFILE_STARTUP", "0") == "1"
        self._start_time = time.perf_counter()

    @contextmanager
    def timer(self, name: str, **metadata):
        if not self.enabled:
            yield
            return
        full_name = f"{self.stack[-1]}.{name}" if self.stack else name
        self.stack.append(full_name)
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.records[full_name] = {
                "duration": end - start,
                "start": start - self._start_time,
                "metadata": metadata,
            }
            self.stack.pop()

    def summary(self) -> str:
        """Human-readable timing breakdown."""
        ...

    def to_chrome_trace(self) -> str:
        """Export as Chrome trace format for chrome://tracing or Perfetto."""
        ...
```

### Usage

```bash
# Enable startup profiling
TRTLLM_PROFILE_STARTUP=1 trtllm-serve meta-llama/Llama-3.1-70B --load-format mx-gms

# Output:
# ================================================================
# TRT-LLM STARTUP TIMING BREAKDOWN
# ================================================================
# config_load: 1.20s (0.3%)
# model_download: 0.00s (0.0%) ← MX: skipped (P2P)
# weight_loading: 0.10s (0.0%) ← GMS: zero-copy import
# weight_processing: 12.50s (18.0%)
# compilation: 52.30s (75.3%) ← Now the dominant bottleneck
# kv_cache_allocation: 2.10s (3.0%)
# executor_init: 1.30s (1.9%)
# server_startup: 0.90s (1.3%)
# ================================================================
# Total startup time: 69.40s
# ================================================================
```

## Benchmark Baselines

```yaml
# tests/benchmarks/startup_baselines.yaml
baselines:
  TinyLlama-1.1B:
    cold_start: 15.0s
    mx_p2p: 5.0s
    gms_ro: 2.0s

  Llama-3.1-8B:
    cold_start: 45.0s
    mx_p2p: 10.0s
    gms_ro: 3.0s

  Llama-3.1-70B:
    cold_start: 300.0s
    mx_p2p: 30.0s
    gms_ro: 5.0s

  DeepSeek-V3-681B:
    cold_start: 600.0s
    mx_p2p: 30.0s
    gms_ro: 5.0s
```

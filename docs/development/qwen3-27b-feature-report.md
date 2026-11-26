# Qwen3.8-27B Support Compared with Qwen3-0.6B

## Target and test scope

The repository's 27B target is [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), the model used by the Runpod benchmark. The offline test configuration is based on revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. It retains the published 5,120-wide hidden state, 17,408-wide feed-forward network, 248,320-token vocabulary, attention dimensions, and FP8 configuration, while changing only the text-layer count and layer-type list to the first two layers. Both retained layers use gated DeltaNet linear attention.

As in the existing Qwen3-0.6B integration tests, the model uses deterministic dummy weights, fixed token IDs, and text-only vLLM inference. The existing 0.6B fixture is a smaller one-layer, 128-wide configuration, not the full 0.6B checkpoint. No pretrained checkpoint accuracy is tested.

## Features and changes

| 27B feature absent from the existing 0.6B test | Repository support added or exercised |
| --- | --- |
| BF16 Gemma-style RMS normalization computes with `1 + weight` in FP32. | The importer normalizes `Tensor.float()` to the corresponding ATen dtype conversion. |
| Gated DeltaNet uses a zero-initialized output, contiguous gate vectors, and flattened head outputs. Its convolution and recurrent caches are owned by vLLM. | The importer normalizes `torch.zeros`, `Tensor.contiguous`, and `Tensor.flatten`. IREE partitioning retains zero allocation in the PyTorch driver so the custom GDN operation can mutate the original buffer. Existing custom-operation handling preserves GDN calls and their ordering; Kashi does not replace the GDN implementation. |
| Block FP8 projections use dynamic activation quantization and 128×128 weight scales. CUDA kernels consume transposed weights and permuted scale buffers. | The importer handles `Tensor.permute` and the `aten.numpy_T` overload packet. The rebuilder decodes PyTorch memory-format values for `contiguous`. Existing vLLM FP8 quantization and scaled-matmul custom operations remain driver boundaries; they are not new Kashi kernels. |
| The target uses untied embedding and LM-head matrices and block-FP8 weights/scales. | Test initialization assigns distinct deterministic weights to each layer, sets positive scales, and initializes the large embeddings and head before vLLM converts them into kernel-specific formats. |

## Parity checks and precision

`python/tests/test_qwen38_integration.py` compares separate compiled and uncompiled vLLM engines for the same prompts, weights, prefill, and decode requests. It exercises FX→MLIR→FX, direct IREE CUDA, and IREE runtime execution; checks both distinct GDN layers and FP8 tensors or packed FP8 Marlin operations; and compares token IDs exactly and finite selected-token log probabilities. CPU coverage includes the unquantized BF16 base architecture and, when supported, block-FP8; the installed CPU FP8 block kernel requires AMX and its compiled FP8 operation.

The fixed dummy-weight seed was chosen to avoid tied top-token scores on both local GPUs: the RTX 5060 Ti and RTX 5050 each had a minimum reference top-two log-probability gap of `0.03125` or greater across the eight generated tokens, for both BF16 reduction settings. The test still checks for near ties rather than assuming this gap on other hardware.

The full-FP32-accumulation comparisons use the existing 0.6B tolerances (`rtol=1e-4`, `atol=1e-5`). The tests also exercise PyTorch's default reduced-precision BF16 reductions. For IREE comparisons under that default, the log-probability relative tolerance is one BF16 epsilon (`0.0078125`); token IDs remain exact. On the local RTX 5050, a representative 5,120-wide BF16 linear operation differed from an FP64 reference in 544 output elements for PyTorch's default reduction versus 5 for IREE. Disabling PyTorch's reduced-precision reduction reduced the PyTorch/IREE disagreement to 4 elements.

The GPU test selects an available BF16-capable NVIDIA device with at least 7 GiB free and enough space for vLLM's 80% memory budget plus a 256 MiB child-context reserve. GPU cases run in isolated processes; use `pytest -s` to see their progress. CPU tests require a CPU vLLM wheel and at least 10 GiB available memory, accounting for the process's cgroup-v2 limits and their ancestors; the CPU FP8 case additionally requires AMX and vLLM's FP8 op. Hardware-dependent cases are explicitly skipped when those resources are absent. IREE execution in this checkout is CUDA-only.

## Coverage limits

The first two layers do not include the full-attention layer that first appears at index 3. Vision/video processing and speculative multi-token prediction are not exercised by these text-only tests. CPU FP8 and CUDA Marlin FP8 were not exercised on the local non-AMX, native-FP8 machine, and the tests do not establish correctness for all 64 layers, pretrained weights, or other hardware/backends.

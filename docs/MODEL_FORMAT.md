# FCPE GGUF v1

`general.architecture = "fcpe"`, `fcpe.version = 1`. Numeric architecture fields are unsigned 32-bit values: `fcpe.sample_rate`, `hop_size`, `n_fft`, `win_size`, `mel_bins`, `hidden_dims`, `layers`, `bins`. The `fcpe.f0_min` and `fcpe.f0_max` fields are float32. Source/checkpoint SHA256 and the original JSON configuration are embedded as strings.

The loader validates metadata types/ranges, the exact tensor count, every tensor shape/type, and all file data ranges before graph execution. Version 1 supports the released conv-only CFNaiveMelPE architecture, with four input normalization groups, 31-tap depthwise kernels, expansion factor two and normalization epsilon 1e-5.

Tensor shapes below use PyTorch / row-major notation; GGUF stores dimension extents reversed for ggml. For example `[out, in, kernel]` becomes ggml `[kernel, in, out]` without transposing the bytes.

| Tensor | Shape | Storage |
| --- | --- | --- |
| `input_stack.0.weight` | `[512,128,3]` | F32/F16 |
| `input_stack.0.bias` | `[512]` | F32 |
| `input_stack.1.weight`, `.bias` | `[512]` | F32 |
| `input_stack.3.weight` | `[512,512,3]` | F32/F16 |
| `input_stack.3.bias` | `[512]` | F32 |
| `block.N.norm.weight`, `.bias` | `[512]` | F32 |
| `block.N.in.weight`, `.bias` | `[2048,512]`, `[2048]` | F32/F16 weight, F32 bias |
| `block.N.dw.weight`, `.bias` | `[1024,1,31]`, `[1024]` | F32 |
| `block.N.out.weight`, `.bias` | `[512,1024]`, `[512]` | F32/F16 weight, F32 bias |
| `norm.weight`, `.bias` | `[512]` | F32 |
| `output_proj.weight`, `.bias` | `[360,512]`, `[360]` | F32/F16 weight, F32 bias |
| `cent_table` | `[360]` | F32 |
| `mel.basis` | `[128,513]` | F32 |
| `mel.window` | `[1024]` | F32 |

`N` is 0..5. Pointwise convolution singleton kernel dimensions are removed. The converter fuses `output_proj.weight_g/weight_v` (or the parametrization equivalents) using PyTorch's weight-norm operation. Training-only gaussian-mask and unused attention-normalization tensors are excluded.

The graph is:

```text
Mel [T,128]
  -> Conv1d(3) -> GroupNorm(4, across channels AND time) -> LeakyReLU(0.01)
  -> Conv1d(3)
  -> 6 x residual[LayerNorm -> Linear(512,2048) -> GLU
                   -> DepthwiseConv1d(31) -> SiLU -> Linear(1024,512)]
  -> LayerNorm -> Linear(512,360) -> sigmoid
  -> CPU cents-weighted decoding -> F0 / confidence / UV
```

Frontend: periodic Hann window; explicit reflect padding (constant padding for short inputs); unnormalized real FFT; `sqrt(real² + imag² + 1e-9)`; Slaney-normalized Mel basis; `log(max(mel, 1e-5))`; duplicate the last frame to reach `floor(resampled_samples / hop) + 1` frames. Stored basis/window values are generated using the same float32 conventions as the release.

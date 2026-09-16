# MLX inference components

`dflash_model.py` is adapted from ARahim3/mlx-dspark commit
`d2719285af03e875efbb540c459c5971947ac015`, which ports z-lab/dflash's
MLX drafter and Inco AI's DFlash 2 components from SGLang.
The original MIT notice for Z Lab is retained in the file. DFlash 2 components
are also subject to Apache License 2.0, reproduced in LICENSE-APACHE-2.0.txt.
The prefix-cache class is omitted; Geist supplies its own generation loop.

`qwen_small_m.py` adapts the Metal kernel from avlp12/mlx-lm
`mlx_lm/fast_qmm.py` through the same mlx-dspark commit.
Geist uses instance-specific dispatch instead of global class patching.
Its direct-fragment kernel uses the fragment-coordinate mapping from MLX Steel's
`mlx/backend/metal/kernels/steel/gemm/mma.h`, Copyright © 2024 Apple Inc., MIT.

`qwen_speculative.py` follows the Qwen 3.5 forward implementation in
ml-explore/mlx-lm 0.31.3, Copyright © 2026 Apple Inc., MIT License.

## MIT License

Copyright (c) 2026 erahim3, Z Lab, the fast_qmm.py authors, and Apple Inc.
Copyright © 2024 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

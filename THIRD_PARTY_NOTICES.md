# Third-party notices

The nugget judge prompt in `src/goodmem_beam/prompts/nugget_judge.txt` is copied
unchanged from `src/prompts.py` in the official BEAM repository at commit
`b2da22eac88bb0874c64665f13457eb99835774a`.

Event equivalence and ranking semantics follow that repository's
`src/evaluation/compute_metrics.py`; aggregation follows
`src/evaluation/report_results.py`.

Source: https://github.com/mohammadtavakoli78/BEAM

## Upstream MIT license

MIT License

Copyright (c) 2025 Mohammad Tavakoli

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


## Dataset

BEAM-10M by Mohammad Tavakoli et al. is distributed separately under
Creative Commons Attribution-ShareAlike 4.0. Download from
https://huggingface.co/datasets/Mohammadta/BEAM-10M .
Dataset-derived evaluation artifacts in `results/` retain CC BY-SA 4.0;
the harness code is MIT licensed. The source conversations are not distributed here.

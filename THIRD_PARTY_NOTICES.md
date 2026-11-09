# 第三者の著作権表示

本リポジトリには、下記のソフトウェアを Rust に移植したコードが含まれる。

モデルの重みは本リポジトリに含まれない。重みの利用条件は配布元の定めに従う。

## Music Source Separation Training

- URL: https://github.com/ZFTurbo/Music-Source-Separation-Training
- 対象: タグ `v1.0.23`（コミット `f17924ddb1817cdbae6353706708c4518dd7cd14`）
- 移植したコード:
  - `models/bs_roformer/bs_roformer.py` → `src/model/mod.rs`、`src/model/attention.rs`、`src/model/band.rs`、`src/model/normalization.rs`、`src/separator.rs`
  - `models/bs_roformer/attend.py` → `src/model/attention.rs`
  - `utils/model_utils.py` の `demix` と `_getWindowingArray` → `src/demix/`

```text
MIT License

Copyright (c) 2024 Roman Solovyev (ZFTurbo)

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
```

## BS-RoFormer

- URL: https://github.com/lucidrains/BS-RoFormer
- 対象: コミット `4b951924b57f46f817fc80ac17b828c7af01094d` で確認した `LICENSE`。`LICENSE` は `390cf293367cba1051f12b68daf549bd4b12e647` で追加された後、このコミットまで変更されていない。
- 移植したコード: `bs_roformer/bs_roformer.py` と `bs_roformer/attend.py`。Music Source Separation Training の `models/bs_roformer/bs_roformer.py` と `models/bs_roformer/attend.py` は、この 2 ファイルに由来する。Music Source Separation Training の `README.md` に BS-RoFormer の実装元として lucidrains/BS-RoFormer が記載されている。Rust の移植先は Music Source Separation Training の節に記載した同名ファイルの移植先と同一である。

```text
MIT License

Copyright (c) 2023 Phil Wang

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
```

## rotary-embedding-torch

- URL: https://github.com/lucidrains/rotary-embedding-torch
- 対象: タグ `0.3.5`（コミット `2cab6c4d94edabcbc00e450477b88ce22ea73bf2`）
- 移植したコード: `rotary_embedding_torch/rotary_embedding_torch.py` の回転位置埋め込み → `src/model/rotary/`

```text
MIT License

Copyright (c) 2021 Phil Wang

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
```

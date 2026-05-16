# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download GSM8K from HuggingFace and dump as parquet at ``~/data/gsm8k/``.

Used by Phase J e2e (``run_phase_j_e2e.sh``) which expects the data in this
location. Skip if files already present.

Run:
    python tests/experimental/tree_training/download_gsm8k.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(f"`datasets` not installed in current env: {exc}")
        return 1

    out_dir = os.path.expanduser("~/data/gsm8k")
    os.makedirs(out_dir, exist_ok=True)

    for split in ("train", "test"):
        out_path = os.path.join(out_dir, f"{split}.parquet")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue
        print(f"downloading openai/gsm8k {split} ...")
        ds = load_dataset("openai/gsm8k", "main", split=split)
        ds.to_parquet(out_path)
        print(f"[ok] wrote {out_path} ({len(ds)} rows)")

    print(f"\nGSM8K ready at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

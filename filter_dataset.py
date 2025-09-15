#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a two-split (train+test) preference dataset for SimPO by:
  1) Filtering UltraFeedback by `source` (e.g., "false_qa")
  2) Keeping only ARMORM rows whose `prompt` matches those instructions
  3) Exporting prompt/chosen/rejected and pushing a DatasetDict({'train', 'test'}) to the Hub
"""

import argparse
import re
from typing import Any, Dict, Set

from datasets import load_dataset, Dataset, DatasetDict, get_dataset_split_names

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def as_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("text", "content", "response", "output"):
            if k in v and isinstance(v[k], str):
                return v[k]
    if isinstance(v, (list, tuple)):
        for el in v:
            if isinstance(el, str):
                return el
            if isinstance(el, dict):
                for k in ("text", "content", "response", "output"):
                    if k in el and isinstance(el[k], str):
                        return el[k]
        return "" if not v else str(v[0])
    return str(v)

def filter_and_prepare_split(arm_split: str, instruction_keys: Set[str], num_proc: int) -> Dataset:
    arm = load_dataset("princeton-nlp/gemma2-ultrafeedback-armorm", split=arm_split)
    arm = arm.filter(lambda ex: normalize_text(ex["prompt"]) in instruction_keys,
                     num_proc=num_proc)
    # arm = arm.map(
    #     lambda ex: {
    #         "prompt": ex["prompt"],
    #         "chosen": as_text(ex["chosen"]),
    #         "rejected": as_text(ex["rejected"])
    #     },
    #     remove_columns=[c for c in arm.column_names if c not in {"prompt","chosen","rejected"}],
    #     num_proc=num_proc
    # )
    # arm = arm.filter(lambda ex: bool(ex["prompt"]) and bool(ex["chosen"]) and bool(ex["rejected"]),
    #                  num_proc=num_proc)
    return arm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uf_source", default="false_qa", help='UltraFeedback "source" tag (e.g., "false_qa").')
    ap.add_argument("--repo_id",  default="Alligator123/gemma2-ultrafeedback-armorm-false_qa", help="HF Hub repo id to push to, e.g., user/my-dataset")
    ap.add_argument("--private", action="store_true", help="Push as a private dataset")
    ap.add_argument("--num_proc", type=int, default=8)
    args = ap.parse_args()

    # 1) Build instruction set from UltraFeedback[source==uf_source]
    uf = load_dataset("openbmb/UltraFeedback", split="train")
    uf_sub = uf.filter(lambda ex: ex.get("source") == args.uf_source, num_proc=args.num_proc)
    instruction_keys = {normalize_text(t) for t in uf_sub["instruction"]}
    print(f"[UltraFeedback] source={args.uf_source!r}: {len(uf_sub):,} rows; unique instructions={len(instruction_keys):,}")

    # 2) Prepare train + test splits of ARMORM
    armorm_splits = set(get_dataset_split_names("princeton-nlp/gemma2-ultrafeedback-armorm"))
    out_dict: Dict[str, Dataset] = {}

    if "train" in armorm_splits:
        train_ds = filter_and_prepare_split("train", instruction_keys, args.num_proc)
        print(f"[ARMORM] train kept: {len(train_ds):,}")
        out_dict["train"] = train_ds

    if "test" in armorm_splits:
        test_ds = filter_and_prepare_split("test", instruction_keys, args.num_proc)
        print(f"[ARMORM] test kept:  {len(test_ds):,}")
        out_dict["test"] = test_ds

    assert out_dict, "No splits produced; aborting."

    # 3) Push to Hub
    dsd = DatasetDict(out_dict)
    dsd.push_to_hub(args.repo_id, private=args.private,
                    commit_message=f"Add {args.uf_source} filtered ARMORM (prompt/chosen/rejected)")
    print(f"[OK] Pushed to hub: https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()

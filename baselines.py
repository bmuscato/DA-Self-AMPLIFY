from __future__ import annotations


import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_ent

from shared import (
    SLM, load_dataset, parse_annotations, set_task, get_task_config,
    build_zero_shot_messages, build_io_messages, build_rationale_messages,
    predict_dataset,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def select_random_shots(df_train, n_per_class=10, seed=42):
    
    rng = np.random.default_rng(seed)
    indices = []
    for cls in sorted(df_train["hard_label"].unique()):  # 0 first, then 1
        pool = df_train[df_train["hard_label"] == cls].index.tolist()
        n = min(n_per_class, len(pool))
        indices.extend(list(rng.choice(pool, size=n, replace=False)))
    logger.info("Selected %d random shots (%d per class, ordered: neg→pos)",
                len(indices), n_per_class)
    return indices


def load_shots_from_csv(shots_path, shots_per_class=None):
    
    df_shots = pd.read_csv(shots_path)
    logger.info("Loaded %d shots from %s", len(df_shots), shots_path)

    if shots_per_class is not None:
        selected = []
        for label in sorted(df_shots["hard_label"].unique()):
            class_rows = df_shots[df_shots["hard_label"] == label]
            selected.append(class_rows.head(shots_per_class))
            logger.info("    Class %d: %d available → selected %d",
                         label, len(class_rows), min(shots_per_class, len(class_rows)))
        df_shots = pd.concat(selected)
        logger.info("  Stratified selection: %d shots (%d per class)",
                     len(df_shots), shots_per_class)

    shots = []
    for _, row in df_shots.iterrows():
        shots.append({
            "text": str(row["text"]),
            "label": int(row["hard_label"]),
        })

    labels = [s["label"] for s in shots]
    logger.info("  Shot labels: %s", labels)
    return shots


def generate_auto_cot(slm, text, label, task=None):
    
    cfg = get_task_config(task)

    messages = [
        {"role": "system", "content": cfg["auto_cot_system"]},
        {"role": "user", "content": cfg["auto_cot_user"].format(text=text)},
    ]
    reasoning = slm.generate(messages, max_new=80)

   
    split_word = cfg["cot_split_word"]
    reasoning = reasoning.split(split_word)[0].strip().rstrip(".")

    
    conclusion = cfg["cot_conclusion"].format(label=label)
    return f"{reasoning}. {conclusion}"


def run_step1(train_path, test_path, model_name, n_per_class=10,
              shots_per_class=None, device="auto", output_dir=".",
              condition="all", shots_path=None, task="offensive_en"):

   
    set_task(task)

    
    if shots_per_class is None:
        shots_per_class = n_per_class

    name = Path(train_path).stem.replace("train_", "").replace("_train", "")
    model_short = model_name.split("/")[-1]

    if shots_path:
        shot_name = Path(shots_path).stem
    else:
        shot_name = "random"

    logger.info("=" * 64)
    logger.info("  STEP 1 — %s + %s + %s + task=%s", name, model_short, shot_name, task)
    logger.info("  Shots per class: %d (total: %d)", shots_per_class, shots_per_class * 2)
    logger.info("=" * 64)

    df_train = load_dataset(train_path)
    df_test = load_dataset(test_path)

   
    df_test["_anns"] = df_test["annotations"].apply(parse_annotations)
    df_test["mace_entropy"] = df_test["_anns"].apply(
        lambda a: float(sp_ent(np.bincount(a, minlength=2) / len(a), base=2)))
    df_test.drop(columns=["_anns"], inplace=True)

    slm = SLM(model_name, device)

   
    if shots_path:
        io_shots = load_shots_from_csv(shots_path, shots_per_class=shots_per_class)
    else:
        shot_idx = select_random_shots(df_train, shots_per_class)
        io_shots = [{"text": str(df_train.loc[i, "text"]),
                     "label": int(df_train.loc[i, "hard_label"])}
                    for i in shot_idx]

    spc_tag = f"{shots_per_class}spc"
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    #zero-shot 
    if condition in ("all", "zero_shot"):
        logger.info("\n  [Zero-shot]")
        predict_dataset(
            slm, df_test,
            lambda txt: build_zero_shot_messages(txt, task=task),
            "zero_shot",
            str(out_dir / f"preds_zero_shot_{name}_{model_short}.csv"),
            task=task)

    #IO (input-output) 
    if condition in ("all", "IO"):
        logger.info("\n  [IO — %s, %d shots (%d/class), (x, y)]",
                     shot_name, len(io_shots), shots_per_class)
        predict_dataset(
            slm, df_test,
            lambda txt: build_io_messages(txt, io_shots, task=task),
            f"IO_{shot_name}_{spc_tag}",
            str(out_dir / f"preds_IO_{shot_name}_{spc_tag}_{name}_{model_short}.csv"),
            task=task)

    #Auto-CoT
    if condition in ("all", "Auto-CoT"):
        logger.info("\n  [Auto-CoT — %s, %d shots (%d/class), generating rationales...]",
                     shot_name, len(io_shots), shots_per_class)
        auto_cot_shots = []
        for shot in io_shots:
            rationale = generate_auto_cot(slm, shot["text"], shot["label"], task=task)
            auto_cot_shots.append({
                "text": shot["text"],
                "label": shot["label"],
                "rationale": rationale,
            })
            logger.info("    → %s", rationale[:70])

        predict_dataset(
            slm, df_test,
            lambda txt: build_rationale_messages(txt, auto_cot_shots, task=task),
            f"Auto-CoT_{shot_name}_{spc_tag}",
            str(out_dir / f"preds_Auto-CoT_{shot_name}_{spc_tag}_{name}_{model_short}.csv"),
            task=task)

    logger.info("\n  Done. CSVs in %s/", out_dir)
    logger.info("=" * 64)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 1: Baselines")
    p.add_argument("--train", required=True, help="Train CSV")
    p.add_argument("--test", required=True, help="Test CSV")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--task", default="offensive_en",
                   choices=["offensive_en", "irony_it"],
                   help="Task/language: offensive_en or irony_it")
    p.add_argument("--condition", default="all",
                   choices=["all", "zero_shot", "IO", "Auto-CoT"])
    p.add_argument("--shots", default=None,
                   help="Shots CSV from step2 (e.g. shots_difficult_ordered.csv). "
                        "If not provided, uses random selection.")
    p.add_argument("--n_per_class", type=int, default=10,
                   help="Shots per class for random selection (ignored if --shots is given)")
    p.add_argument("--shots_per_class", type=int, default=None,
                   help="Max shots per class to use (applies to both random and CSV shots). "
                        "Defaults to --n_per_class if not set.")
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default=".")
    args = p.parse_args()
    run_step1(args.train, args.test, args.model, args.n_per_class,
              args.shots_per_class, args.device, args.output_dir,
              args.condition, args.shots, args.task)
from __future__ import annotations


import argparse
import subprocess
import sys
import tempfile
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from scipy import stats as sp_stats

from shared import parse_annotations

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def build_mace_csv(df):
    all_anns = [parse_annotations(r) for r in df["annotations"]]
    mx = max(len(a) for a in all_anns)
    lines = []
    for anns in all_anns:
        row = [str(anns[i]) if i < len(anns) else "" for i in range(mx)]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def run_mace(mace_input, mace_path, iterations=50, restarts=10):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.csv").write_text(mace_input)
        pfx = str(tmp / "out")
        cmd = [sys.executable, mace_path, "--entropies", "--prefix", pfx,
               "--iterations", str(iterations), "--restarts", str(restarts),
               str(tmp / "in.csv")]
        logger.info("Running MACE: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"MACE failed: {result.stderr[:300]}")
        ent_f = Path(pfx + ".entropies")
        comp_f = Path(pfx + ".competence")
        entropies = ([float(x) for x in ent_f.read_text().strip().split("\n")
                      if x.strip()] if ent_f.exists() else [])
        competences = ([float(x) for x in comp_f.read_text().strip().split("\t")
                        if x.strip()] if comp_f.exists() else [])
    return entropies, competences



def select_ambiguous(class_df, n, seed):
    
    med = class_df["mace_entropy"].median()
    mad = sp_stats.median_abs_deviation(class_df["mace_entropy"], scale=1)
    pool = class_df[class_df["mace_entropy"].between(med - mad, med + mad)]

    if len(pool) >= n:
        logger.info("    MAD window: med=%.6f MAD=%.6f pool=%d → sampling %d",
                     med, mad, len(pool), n)
        return pool.sample(n=n, random_state=seed)

    
    logger.warning("    MAD window too narrow (med=%.6f MAD=%.6f pool=%d, need=%d). "
                   "Falling back to closest-to-median.", med, mad, len(pool), n)
    class_df = class_df.copy()
    class_df["_dist_to_med"] = (class_df["mace_entropy"] - med).abs()
    closest = class_df.nsmallest(n, "_dist_to_med").drop(columns=["_dist_to_med"])
    return closest


def select_and_save_shots(df, n_per_class=10, seed=42, output_dir=".", dataset_name="dataset"):
    
    out_dir = Path(output_dir)

    n0 = min(n_per_class, len(df[df["hard_label"] == 0]))
    n1 = min(n_per_class, len(df[df["hard_label"] == 1]))

    #difficult
    class_0_desc = df[df["hard_label"] == 0].sort_values("mace_entropy", ascending=False)
    class_1_desc = df[df["hard_label"] == 1].sort_values("mace_entropy", ascending=False)

    diff_0 = class_0_desc.iloc[:n0]
    diff_1 = class_1_desc.iloc[:n1]

    diff_ordered = pd.concat([diff_0, diff_1])
    diff_ordered.to_csv(out_dir / f"{dataset_name}_difficult_ordered.csv",
                        columns=["text", "hard_label"], index=False)

    diff_shuffled = pd.concat([diff_0, diff_1]).sample(frac=1, random_state=seed)
    diff_shuffled.to_csv(out_dir / f"{dataset_name}_difficult_shuffled.csv",
                         columns=["text", "hard_label"], index=False)

    logger.info("  DIFFICULT: %d class-0 (H: %.4f→%.4f) + %d class-1 (H: %.4f→%.4f) = %d total",
                len(diff_0), diff_0["mace_entropy"].max(), diff_0["mace_entropy"].min(),
                len(diff_1), diff_1["mace_entropy"].max(), diff_1["mace_entropy"].min(),
                len(diff_0) + len(diff_1))

    #ambiguous
    logger.info("  AMBIGUOUS selection:")
    logger.info("    Class 0:")
    amb_0 = select_ambiguous(class_0_desc, n0, seed)
    logger.info("    Class 1:")
    amb_1 = select_ambiguous(class_1_desc, n1, seed)

    amb_ordered = pd.concat([amb_0, amb_1])
    amb_ordered.to_csv(out_dir / f"{dataset_name}_ambiguous_ordered.csv",
                       columns=["text", "hard_label"], index=False)

    amb_shuffled = pd.concat([amb_0, amb_1]).sample(frac=1, random_state=seed)
    amb_shuffled.to_csv(out_dir / f"{dataset_name}_ambiguous_shuffled.csv",
                        columns=["text", "hard_label"], index=False)

    logger.info("  AMBIGUOUS: %d class-0 + %d class-1 = %d total",
                len(amb_0), len(amb_1), len(amb_0) + len(amb_1))

    #easy 
    class_0_asc = df[df["hard_label"] == 0].sort_values("mace_entropy", ascending=True)
    class_1_asc = df[df["hard_label"] == 1].sort_values("mace_entropy", ascending=True)

    easy_0 = class_0_asc.iloc[:n0]
    easy_1 = class_1_asc.iloc[:n1]

    easy_ordered = pd.concat([easy_0, easy_1])
    easy_ordered.to_csv(out_dir / f"{dataset_name}_easy_ordered.csv",
                        columns=["text", "hard_label"], index=False)

    easy_shuffled = pd.concat([easy_0, easy_1]).sample(frac=1, random_state=seed)
    easy_shuffled.to_csv(out_dir / f"{dataset_name}_easy_shuffled.csv",
                         columns=["text", "hard_label"], index=False)

    logger.info("  EASY: %d class-0 (H: %.4f→%.4f) + %d class-1 (H: %.4f→%.4f) = %d total",
                len(easy_0), easy_0["mace_entropy"].min(), easy_0["mace_entropy"].max(),
                len(easy_1), easy_1["mace_entropy"].min(), easy_1["mace_entropy"].max(),
                len(easy_0) + len(easy_1))


def run_step2(data_path, mace_path=None, n_per_class=10, output_dir="."):
    name = Path(data_path).stem
    logger.info("=" * 64)
    logger.info("  STEP 2 — MACE entropy + shot selection: %s", name)
    logger.info("=" * 64)

    df = pd.read_csv(data_path)
    df["_anns"] = df["annotations"].apply(parse_annotations)
    logger.info("Loaded %d instances", len(df))

  
    used_mace = False
    if mace_path and Path(mace_path).exists():
        try:
            mace_csv = build_mace_csv(df)
            entropies, competences = run_mace(mace_csv, mace_path)
            if len(entropies) == len(df):
                df["mace_entropy"] = entropies
                used_mace = True
                logger.info("  MACE entropy computed")
                if competences:
                    logger.info("  Annotator competences: %s",
                                [f"{c:.3f}" for c in competences])
        except Exception as e:
            logger.warning("  MACE failed (%s)", e)

    

    df.drop(columns=["_anns"], inplace=True)

    
    n_ag = (df["mace_entropy"] == 0).sum()
    n_dis = (df["mace_entropy"] > 0).sum()
    logger.info("\n  Entropy statistics:")
    logger.info("    instances: %d (agreed=%d, disagreed=%d)", len(df), n_ag, n_dis)
    logger.info("    mean=%.4f  median=%.4f  min=%.4f  max=%.4f",
                df["mace_entropy"].mean(), df["mace_entropy"].median(),
                df["mace_entropy"].min(), df["mace_entropy"].max())
    for cls in sorted(df["hard_label"].unique()):
        sub = df[df["hard_label"] == cls]
        logger.info("    class %d: n=%d  mean_H=%.4f  median_H=%.4f",
                     cls, len(sub), sub["mace_entropy"].mean(),
                     sub["mace_entropy"].median())


    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    out_csv = str(out_dir / f"{name}_mace.csv")
    df.to_csv(out_csv, index=False)
    logger.info("\n  Enriched CSV → %s", out_csv)

 
    logger.info("\n  Shot selection (n_per_class=%d):", n_per_class)
    select_and_save_shots(df, n_per_class, output_dir=str(out_dir), dataset_name=name)

    logger.info("=" * 64)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 2: MACE preprocessing")
    p.add_argument("--train", required=True)
    p.add_argument("--mace_path", default=None)
    p.add_argument("--n_per_class", type=int, default=10)
    p.add_argument("--output_dir", default=".")
    p.add_argument("--no_mace", action="store_true")
    args = p.parse_args()
    run_step2(args.train,
              mace_path=None if args.no_mace else args.mace_path,
              n_per_class=args.n_per_class,
              output_dir=args.output_dir)

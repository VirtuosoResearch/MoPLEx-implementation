# %%
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset

CRITERIA = ["helpfulness", "truthfulness", "instruction_following", "honesty"]


def extract_all_scores(dataset) -> pd.DataFrame:
    rows = []
    for ex_id, row in enumerate(dataset):
        for comp_idx, comp in enumerate(row["completions"]):
            ann = comp.get("annotations", {})
            entry = {"example_id": ex_id, "comp_idx": comp_idx}

            ok = True
            for crit in CRITERIA:
                crit_ann = ann.get(crit, None)
                if crit_ann is None or crit_ann.get("Rating") is None:
                    ok = False
                    break
                try:
                    entry[crit] = float(crit_ann["Rating"])
                except:
                    ok = False
                    break

            if ok:
                rows.append(entry)

    return pd.DataFrame(rows)


def plot_difference_distributions(df: pd.DataFrame):
    sns.set(style="whitegrid")

    pairs = list(itertools.combinations(CRITERIA, 2))
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()

    for ax, (c1, c2) in zip(axes, pairs):
        diff = df[c1] - df[c2]

        sns.kdeplot(diff, fill=True, alpha=0.4, bw_adjust=0.8, ax=ax)
        ax.axvline(0, linestyle="--", color="black", alpha=0.6)

        ax.set_title(f"{c1} − {c2}")
        ax.set_xlabel("score difference")
        ax.set_ylabel("density")

    plt.tight_layout()
    plt.savefig("criteria_diff_distributions.png", dpi=300)
    plt.close()


def main():
    ds = load_dataset("openbmb/UltraFeedback", split="train")

    df = extract_all_scores(ds)
    df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

    plot_difference_distributions(df)


if __name__ == "__main__":
    main()

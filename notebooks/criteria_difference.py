# %%
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

CRITERIA = ["helpfulness", "truthfulness", "instruction_following", "honesty"]


def extract_all_scores(dataset) -> pd.DataFrame:
    rows = []
    for ex_id, row in tqdm(enumerate(dataset), total=len(dataset)):
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

    for c1, c2 in pairs:
        diff = df[c1] - df[c2]

        fig, ax = plt.subplots(figsize=(4, 3))

        sns.kdeplot(diff, fill=True, alpha=0.4, bw_adjust=0.8, ax=ax)
        ax.axvline(0, linestyle="--", color="black", alpha=0.6)

        # ax.set_title(rf"\text{{{c1}}} - \text{{{c2}}}")
        ax.set_xlabel(r"$Difference$")
        ax.set_ylabel(r"$Density$")

        plt.tight_layout()
        out_name = f"diff_{c1}_{c2}.png"
        plt.savefig(out_name, dpi=300)
        plt.close(fig)


def main():
    ds = load_dataset("openbmb/UltraFeedback", split="train[:5000]")

    df = extract_all_scores(ds)
    # df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

    plot_difference_distributions(df)


if __name__ == "__main__":
    main()

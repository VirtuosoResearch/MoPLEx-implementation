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

# %%
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from matplotlib import rc
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # optional

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
                except Exception:
                    ok = False
                    break

            if ok:
                rows.append(entry)

    return pd.DataFrame(rows)


def plot_3d_scores(df: pd.DataFrame,
                   c1="helpfulness",
                   c2="truthfulness",
                   c3="honesty",
                   out_name="scores_3d_helpfulness_truthfulness_honesty.pdf",
                   max_points=40000,
                   seed=42):
    # optional subsample to keep the plot readable
    if max_points is not None and len(df) > max_points:
        df_plot = df.sample(n=max_points, random_state=seed).reset_index(drop=True)
    else:
        df_plot = df

    x = df_plot[c1].to_numpy()
    y = df_plot[c2].to_numpy()
    z = df_plot[c3].to_numpy()

    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, s=4, alpha=0.15, linewidths=0)

    # Properly format labels for LaTeX (escape underscores)
    label1 = c1.replace("_", r"\_")
    label2 = c2.replace("_", r"\_")
    label3 = c3.replace("_", r"\_")
    ax.set_xlabel(rf"${label1}$")
    ax.set_ylabel(rf"${label2}$")
    ax.set_zlabel(rf"${label3}$")

    # If your ratings are in [1, 5], you can uncomment:
    # ax.set_xlim(1, 5)
    # ax.set_ylim(1, 5)
    # ax.set_zlim(1, 5)

    # better viewing angle
    ax.view_init(elev=18, azim=35)

    # tight_layout doesn't work well with 3D plots, use subplots_adjust instead
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ds = load_dataset("openbmb/UltraFeedback", split="train[:5000]")
    df = extract_all_scores(ds)

    plot_3d_scores(
        df,
        c1="helpfulness",
        c2="truthfulness",
        c3="honesty",
        out_name="scores_3d_helpfulness_truthfulness_honesty.pdf",
        max_points=40000,
    )


if __name__ == "__main__":
    main()


# %%
from pathlib import Path
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "ranking_varying_m.csv"

df = pd.read_csv(csv_path)
df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
df = df.dropna(subset=["Step"]).sort_values("Step")

m_values = [10, 5, 4, 3, 2]
run_template = (
    "small-transformers-n12k-m{m}-k3-train3000-lr5e-5-mstep20"
    " - val_mid/posterior_acc_aligned"
)

plt.figure(figsize=(7.5, 5.5))
smoothing_window = 1  # Average over nearby points to reduce jaggedness.

for m in m_values:
    metric_col = run_template.format(m=m)
    if metric_col not in df.columns:
        print(f"Skipping m={m}: missing column '{metric_col}'")
        continue

    y = pd.to_numeric(df[metric_col], errors="coerce")
    y_smooth = y.rolling(window=smoothing_window, center=True, min_periods=1).mean()
    valid = y_smooth.notna()
    plt.plot(
        df.loc[valid, "Step"],
        y_smooth.loc[valid],
        # marker="o",
        # markersize=2.8,
        linewidth=3,
        label=r"$m={m}$".format(m=m),
    )

plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel(r"$\mathrm{Gradient~update~steps}$", fontsize=28)
plt.ylabel(r"$\mathrm{Accuracy}$", fontsize=28)
# plt.title("Posterior Accuracy vs Step for Different m")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=20, loc="lower right")
plt.tight_layout()
plt.savefig(script_dir / "ranking_accuracy_varying_m.pdf")
plt.show()


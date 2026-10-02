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
csv_path = script_dir / "ranking_varying_m_k_3.csv"

df = pd.read_csv(csv_path)
df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
df = df.dropna(subset=["Step"]).sort_values("Step")

m_values = [6, 5, 4, 3, 2]
run_template = (
    "n100000-m{m}-k3-lr5e-5-mstep20"
    " - val_mid/posterior_acc_aligned"
)

plt.figure(figsize=(6, 5.5))
smoothing_window = 5  # Average over nearby points to reduce jaggedness.

for m in m_values:
    metric_col = run_template.format(m=m)
    if metric_col not in df.columns:
        print(f"Skipping m={m}: missing column '{metric_col}'")
        continue

    y = pd.to_numeric(df[metric_col], errors="coerce")
    print(y.to_numpy()[-1])
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

plt.title(r"$n=100000,~k=3$", fontsize=28)
plt.xticks(np.arange(0, 3001, 500), [r"$0$", r"$5$", r"$10$", r"$15$", r"$20$", r"$25$", r"$30$"],
    fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel(r"$\mathrm{Gradient~update~steps~(\times 10^2)}$", fontsize=28)
plt.ylabel(r"$\mathrm{Accuracy}$", fontsize=28)
# plt.title("Posterior Accuracy vs Step for Different m")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=20, loc="lower right")
plt.tight_layout()
plt.savefig(script_dir / "ranking_accuracy_varying_m_k3.pdf")
plt.show()


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
csv_path = script_dir / "ranking_varying_m_k_2.csv"

df = pd.read_csv(csv_path)
df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
df = df.dropna(subset=["Step"]).sort_values("Step")

m_values = [6, 5, 4, 3, 2]
run_template = (
    "n5000-m{m}-k2-lr5e-5-mstep20"
    " - val_mid/posterior_acc_aligned"
)

plt.figure(figsize=(6, 5.5))
smoothing_window = 5  # Average over nearby points to reduce jaggedness.

for m in m_values:
    metric_col = run_template.format(m=m)
    if metric_col not in df.columns:
        print(f"Skipping m={m}: missing column '{metric_col}'")
        continue

    y = pd.to_numeric(df[metric_col], errors="coerce")
    print(y)
    print("----")
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

plt.title(r"$n=1000000,~k=2$", fontsize=28)
plt.xticks(np.arange(0, 3001, 500), [r"$0$", r"$5$", r"$10$", r"$15$", r"$20$", r"$25$", r"$30$"],
    fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel(r"$\mathrm{Gradient~update~steps~(\times 10^2)}$", fontsize=28)
plt.ylabel(r"$\mathrm{Accuracy}$", fontsize=28)
# plt.title("Posterior Accuracy vs Step for Different m")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=20, loc="lower right")
plt.tight_layout()
plt.savefig(script_dir / "ranking_accuracy_varying_m_k2.pdf")
plt.show()


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
csv_path = script_dir / "ranking_varying_m_k_4.csv"

df = pd.read_csv(csv_path)
df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
df = df.dropna(subset=["Step"]).sort_values("Step")

m_values = [6, 5, 4, 3, 2]
run_template = (
    "n10000-m{m}-k4-lr5e-5-mstep20"
    " - val_mid/posterior_acc_aligned"
)

plt.figure(figsize=(6, 5.5))
smoothing_window = 5  # Average over nearby points to reduce jaggedness.

for m in m_values:
    metric_col = run_template.format(m=m)
    if metric_col not in df.columns:
        print(f"Skipping m={m}: missing column '{metric_col}'")
        continue

    y = pd.to_numeric(df[metric_col], errors="coerce")
    print(y)
    print("----")
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

plt.title(r"$n=1000000,~k=4$", fontsize=28)
plt.xticks(np.arange(0, 3001, 500), [r"$0$", r"$5$", r"$10$", r"$15$", r"$20$", r"$25$", r"$30$"],
    fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel(r"$\mathrm{Gradient~update~steps~(\times 10^2)}$", fontsize=28)
plt.ylabel(r"$\mathrm{Accuracy}$", fontsize=28)
# plt.title("Posterior Accuracy vs Step for Different m")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=20, loc="lower right")
plt.tight_layout()
plt.savefig(script_dir / "ranking_accuracy_varying_m_k4.pdf")
plt.show()

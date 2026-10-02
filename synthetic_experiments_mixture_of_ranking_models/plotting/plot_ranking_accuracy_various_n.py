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

accs = [
    [0.714, 0.755, 0.755, 0.757, 0.768, 0.776, 0.779],
    [0.901, 0.901, 0.901, 0.903, 0.905, 0.908, 0.911],
    [0.947, 0.959, 0.96, 0.962, 0.962, 0.964, 0.964],
    [0.985, 0.988, 0.99, 0.990, 0.99, 0.992, 0.993],
    [0.995, 0.995, 0.998, 0.998, 0.999, 0.999, 0.999],
]

n_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
m_values = [6, 5, 4, 3, 2]

plt.figure(figsize=(6, 5.5))
smoothing_window = 5  # Average over nearby points to reduce jaggedness.

for m in m_values:
    plt.plot(
        np.arange(len(n_values)),
        accs[-m_values.index(m)-1],
        marker="o",
        markersize=10,
        linewidth=3,
        label=r"$m={m}$".format(m=m),
    )

plt.title(r"$k=2$", fontsize=28)
plt.xticks(np.arange(len(n_values)), [r"$1$", r"$2$", r"$5$", r"$10$", r"$20$", r"$50$", r"$100$"],
    fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel(r"$n~(\times 10^3)$", fontsize=28)
plt.ylabel(r"$\mathrm{Accuracy}$", fontsize=28)
# plt.title("Posterior Accuracy vs Step for Different m")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=20, loc="lower right")
plt.tight_layout()
plt.savefig(script_dir / "ranking_accuracy_varying_n_k2.pdf")
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

accs = [
    [0.324, 0.324, 0.324, 0.324, 0.324, 0.324, 0.324],
    [0.548, 0.777, 0.787, 0.81, 0.81, 0.811, 0.823],
    [0.909, 0.923, 0.934, 0.927, 0.927, 0.927, 0.927],
    [0.937, 0.943, 0.957, 0.967, 0.965, 0.968, 0.97],
    [0.992, 0.997, 0.998, 0.998, 0.998, 0.998, 0.998]
]

n_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
m_values = [6, 5, 4, 3, 2]

plt.figure(figsize=(6, 5.5))
smoothing_window = 5  # Average over nearby points to reduce jaggedness.

for m in m_values:
    plt.plot(
        np.arange(len(n_values)),
        accs[-m_values.index(m)-1],
        marker="o",
        markersize=10,
        linewidth=3,
        label=r"$m={m}$".format(m=m),
    )

plt.title(r"$k=3$", fontsize=28)
plt.xticks(np.arange(len(n_values)), [r"$1$", r"$2$", r"$5$", r"$10$", r"$20$", r"$50$", r"$100$"],
    fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel(r"$n~(\times 10^3)$", fontsize=28)
plt.ylabel(r"$\mathrm{Accuracy}$", fontsize=28)
# plt.title("Posterior Accuracy vs Step for Different m")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=20, loc="lower right")
plt.tight_layout()
plt.savefig(script_dir / "ranking_accuracy_varying_n_k3.pdf")
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

accs = [
    [0.234, 0.245, 0.245, 0.245, 0.245, 0.245, 0.245],
    [0.309, 0.311, 0.68, 0.709, 0.715, 0.72, 0.731],
    [0.628, 0.871, 0.878, 0.887, 0.895, 0.897, 0.904],
    [0.948, 0.948, 0.953, 0.954, 0.955, 0.957, 0.957],
    [0.955, 0.971, 0.983, 0.984, 0.984, 0.985, 0.986]
]

n_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
m_values = [6, 5, 4, 3, 2]

plt.figure(figsize=(6, 5.5))
smoothing_window = 5  # Average over nearby points to reduce jaggedness.

for m in m_values:
    plt.plot(
        np.arange(len(n_values)),
        accs[-m_values.index(m)-1],
        marker="o",
        markersize=10,
        linewidth=3,
        label=r"$m={m}$".format(m=m),
    )

plt.title(r"$k=4$", fontsize=28)
plt.xticks(np.arange(len(n_values)), [r"$1$", r"$2$", r"$5$", r"$10$", r"$20$", r"$50$", r"$100$"],
    fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel(r"$n~(\times 10^3)$", fontsize=28)
plt.ylabel(r"$\mathrm{Accuracy}$", fontsize=28)
# plt.title("Posterior Accuracy vs Step for Different m")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=20, loc="lower right")
plt.tight_layout()
plt.savefig(script_dir / "ranking_accuracy_varying_n_k4.pdf")
plt.show()

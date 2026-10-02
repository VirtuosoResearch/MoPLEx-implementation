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

x = [2, 3, 4, 5, 6, 7, 8]
k_2 = [0.765, 0.901, 0.961, 0.990, 0.995, 0.998, 1.0]
k_3 = [0.294, 0.823, 0.931, 0.967, 0.994, 0.998, 1.0]
k_4 = [0.258, 0.720, 0.839, 0.904, 0.953, 0.978, 0.996]

plt.figure(figsize=(5.5, 4.5))
plt.plot(x, k_2, marker='o', linewidth=5, label=r'$k=2$', markersize=15, color='lightgreen')
plt.plot(x, k_3, marker='s', linewidth=5, label=r'$k=3$', markersize=15, color='royalblue')
plt.plot(x, k_4, marker='^', linewidth=5, label=r'$k=4$', markersize=15, color='darkorange')
plt.xticks(x, [r'$2$', r'$3$', r'$4$', r'$5$', r'$6$', r'$7$', r'$8$'], fontsize=32)
plt.yticks(fontsize=32)
plt.xlabel(r'$m$', fontsize=32)
plt.ylabel(r'$\mathrm{Clustering~Accuracy}$', fontsize=32)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=32)
plt.ylim(0.2, 1.1)
plt.legend(fontsize=28, loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("./clustering_accuracy_aggregated.pdf")

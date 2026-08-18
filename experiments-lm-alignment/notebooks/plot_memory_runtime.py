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

x = [2, 8, 14, 20, 26, 32]
full = [7.605203613, 28.96881152, 37.03090576, 46.09299991]
a_2 = [6.946395898, 10.29120107, 13.01818086, 15.74516064, 18.47214043, 21.19912021]
a_4 = [6.946395898,	13.02004912, 15.68078037, 18.34151162, 21.00224287, 23.66297412]
a_6 = [6.946395898, 15.1529002, 19.88320352, 24.61350684, 29.34381016, 34.07411348]

plt.figure(figsize=(6, 6))
plt.plot(x[:4], full[:4], marker='o', linewidth=5, label=r'$\mathrm{Full~training}$', markersize=15, color='red')
# plt.plot(x[2:4], full[2:4], marker='o', linewidth=5, markersize=15, color='red', linestyle='dashed')
plt.plot(x, a_6, marker='^', linewidth=5, label=r'$\mathrm{MoPLEx}~(a=6)$', markersize=15, color='orange')
plt.plot(x, a_4, marker='s', linewidth=5, label=r'$\mathrm{MoPLEx}~(a=4)$', markersize=15, color='royalblue')
plt.plot(x, a_2, marker='o', linewidth=5, label=r'$\mathrm{MoPLEx}~(a=2)$', markersize=15, color='forestgreen')

# plot a horizontal dashed line at y=48
plt.axhline(y=48, color='gray', linestyle='dashed', linewidth=4)
# write a text at this line
plt.text(2, 50, r'$\mathrm{GPU~Memory~Limit}$', fontsize=36, color='gray')

plt.xticks(x,  fontsize=36)
plt.yticks(fontsize=36)
plt.xlabel(r'$m$', fontsize=36)
plt.ylabel(r'$\mathrm{GPU~Memory~(GB)}$', fontsize=36)
plt.yticks([0, 10, 20, 30, 40, 48], fontsize=36)
plt.ylim(0, 65)
# plt.legend(fontsize=24, frameon=True, loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/plot_memory_cost_varying_m_and_a.pdf")

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

x = [2, 8, 14, 20, 26, 32]
full = [1.310555556, 1.628555556, 2.140277778, 2.852777778]
a_2 = [1.293895556, 1.323455556, 1.329218889, 1.39597, 1.441642778, 1.487315556]
a_4 = [1.314498889, 1.407512222, 1.432285556, 1.500344444, 1.56433, 1.628315556]
a_6 = [1.307783333, 1.534752222, 1.574547778, 1.609871111, 1.665266111, 1.730661111]

plt.figure(figsize=(6, 6))
plt.plot(x[:4], full[:4], marker='o', linewidth=5, label=r'$\mathrm{Full~training}$', markersize=15, color='red')
# plt.plot(x[2:4], full[2:4], marker='o', linewidth=5, markersize=15, color='red', linestyle='dashed')
plt.plot(x, a_6, marker='^', linewidth=5, label=r'$\mathrm{MoPLEx}~(a=6)$', markersize=15, color='orange')
plt.plot(x, a_4, marker='s', linewidth=5, label=r'$\mathrm{MoPLEx}~(a=4)$', markersize=15, color='royalblue')
plt.plot(x, a_2, marker='o', linewidth=5, label=r'$\mathrm{MoPLEx}~(a=2)$', markersize=15, color='forestgreen')

plt.xticks(x,  fontsize=36)
plt.yticks(fontsize=36)
plt.xlabel(r'$m$', fontsize=36)
plt.ylabel(r'$\mathrm{GPU~hours}$', fontsize=36)
plt.yticks(np.arange(1, 4.1, 1), fontsize=36)
plt.ylim(1, 5.5)
plt.legend(fontsize=28, frameon=True, loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/plot_runtime_cost_varying_m_and_a.pdf")

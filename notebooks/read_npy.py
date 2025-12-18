# %%
import numpy as np
t = np.load("../pre_compute/T_Llama-3.2-1B_200.npy")
print(t.shape)
for i in range(5):
    for j in range(5):
            print(i, j, t[i, j])

# %%
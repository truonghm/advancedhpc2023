# %% [markdown]
# - Importing the `numba` package:

# %%
import numba
from numba import cuda 

# %% [markdown]
# - Get GPU information by using `numba.cuda.detect()`: There's only one GPU on this computer.

# %%
print(numba.cuda.detect())

# %% [markdown]
# - Select the GPU and print its name and id:

# %%
gpu = numba.cuda.select_device(0)

print("GPU ID", gpu.id)
print("GPU Name", gpu.name)

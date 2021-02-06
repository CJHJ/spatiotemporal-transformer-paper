# %%
# Load moving mnist raw data
from kedro.extras.datasets.pickle import PickleDataSet
import numpy as np
import pickle as pkl
import seaborn as sns

moving_mnist_data = np.load("../data/01_raw/mnist_test_seq.npy")
moving_mnist_data = moving_mnist_data.transpose((1, 0, 2, 3))
print(moving_mnist_data.shape)

# %%
# Convert to pickle and dump in the same folder
pkl.dump(
    moving_mnist_data,
    open("../data/01_raw/mnist_test_seq.pkl", "wb"),
    pkl.HIGHEST_PROTOCOL,
)
test = PickleDataSet(filepath="../data/01_raw/mnist_test_seq.pkl")
test = test.load()

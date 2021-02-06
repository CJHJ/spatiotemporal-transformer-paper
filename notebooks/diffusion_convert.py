# %%
from pathlib import Path
import os
import pickle as pkl
import re
import torch

PATH = Path("../data/01_raw/diffusion/")
train_path = PATH / "train" / "train"
valid_path = PATH / "valid"

train_data_list = [
    data_filename
    for data_filename in os.listdir(train_path)
    if data_filename[-4:] == ".pkl"
]
train_data_list.sort(key=lambda f: int(re.sub("\D", "", f)))

train_data_list

# %%
def create_pickle_data_from_pickle_files(data_path: Path, pickle_name: str):
    data_list = [
        data_filename
        for data_filename in os.listdir(data_path)
        if data_filename[-4:] == ".pkl"
    ]
    data_list.sort(key=lambda f: int(re.sub("\D", "", f)))

    data = []
    data_latent = []
    data_observation = []
    for i in data_list:
        pickled_data = pkl.load(open(train_path / train_data_list[0], "rb"))
        data.append(data)
        data_latent.append(pickled_data["latent"])
        data_observation.append(pickled_data["observation"])

    data_latent = torch.stack(data_latent).numpy()
    data_observation = torch.stack(data_observation).numpy()

    # Pickling
    prefix_name = pickle_name + "_"
    pkl.dump(data, open(PATH / (prefix_name + "data.pkl"), "wb"))
    pkl.dump(data_latent, open(PATH / (prefix_name + "latent.pkl"), "wb"))
    pkl.dump(
        data_observation, open(PATH / (prefix_name + "observation.pkl"), "wb"),
    )

    return data, data_latent, data_observation


# %%
train_data = create_pickle_data_from_pickle_files(
    train_path, "diffusion_train"
)
valid_data = create_pickle_data_from_pickle_files(
    valid_path, "diffusion_valid"
)


# %%
train_data = pkl.load(open(PATH / "diffusion_train_observation.pkl", "rb"))
valid_data = pkl.load(open(PATH / "diffusion_valid_observation.pkl", "rb"))


# %%
import numpy as np
import seaborn as sns

np.max(train_data), np.min(train_data)


# %%
sns.distplot(train_data.flatten()[:1000])

# %%

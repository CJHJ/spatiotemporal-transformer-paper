# %%
import pandas as pd
from collections import defaultdict
import pickle
from typing import DefaultDict

cmap_data = pickle.load(open("./cmap_transformer.pkl", "rb"))
mm_data = pickle.load(open("./mm_report_transformer.pkl", "rb"))

# %%
def convert_to_metric_first(data):
    rows = defaultdict(dict)

    for model, metrics in data.items():
        for metric, values in metrics.items():
            for i, value in enumerate(values):
                rows[metric][model + f"_{i}"] = value

    return rows


def save_to_csv(data, save_path):
    df = pd.DataFrame(data)
    df.to_csv(save_path)


save_to_csv(
    convert_to_metric_first(cmap_data), "./cmap_report_transformer.csv"
)
save_to_csv(convert_to_metric_first(mm_data), "./mm_report_transformer.csv")

# %%

# %%
import os
from pathlib import Path
from mlflow.tracking import MlflowClient

experiment_id = "8"
tracking_uri = "file:///home/kawa/Projects/spatiotemporal-transformer/mlruns/"

# %%
def save_models_from_experiment_id(
    tracking_uri: str, experiment_id: str, save_path: Path
):
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(experiment_id)

    for run in runs:
        # Get attributes
        model_type = run.data.params["model_type"]
        quantiles = run.data.params["optim_params.quantiles"]
        run_id = run.info.run_id

        # Create folder
        model_name = "quantiles_" if len(quantiles) > 2 else ""
        model_name = model_name + model_type
        model_number = 0
        modif_save_path = save_path / (model_name + f"_{model_number}")
        while os.path.exists(modif_save_path):
            model_number += 1
            modif_save_path = save_path / (model_name + f"_{model_number}")
        os.mkdir(modif_save_path)

        # Download artifacts
        local_path = client.download_artifacts(run_id, ".", modif_save_path)
        print("Artifacts downloaded in: {}".format(local_path))
        print("Artifacts: {}".format(os.listdir(local_path)))


# %%
save_models_from_experiment_id(
    tracking_uri, experiment_id, Path("../../models/MovingMNIST")
)

# %%

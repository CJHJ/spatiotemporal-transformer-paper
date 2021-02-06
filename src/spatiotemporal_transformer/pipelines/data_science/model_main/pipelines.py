from kedro.pipeline import node, Pipeline
from spatiotemporal_transformer.pipelines.data_science.model_main.data_utils_nodes import (
    create_dataloader,
    create_dataset,
)
from spatiotemporal_transformer.pipelines.data_science.model_main.main_model_nodes import (
    prepare_model,
    train_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_dataset,
                inputs=["norm_train_data", "params:dataset_parameters"],
                outputs="train_dataset",
                name="create_train_dataset",
            ),
            node(
                func=create_dataset,
                inputs=["norm_valid_data", "params:dataset_parameters"],
                outputs="valid_dataset",
                name="create_valid_dataset",
            ),
            node(
                func=create_dataset,
                inputs=["norm_test_data", "params:dataset_parameters"],
                outputs="test_dataset",
                name="create_test_dataset",
            ),
            node(
                func=create_dataloader,
                inputs=["train_dataset", "params:train_data_parameters"],
                outputs="train_dataloader",
                name="create_train_dataloader",
            ),
            node(
                func=create_dataloader,
                inputs=["valid_dataset", "params:valid_data_parameters"],
                outputs="valid_dataloader",
                name="create_valid_dataloader",
            ),
            node(
                func=create_dataloader,
                inputs=["test_dataset", "params:test_data_parameters"],
                outputs="test_dataloader",
                name="create_test_dataloader",
            ),
            node(
                func=train_model,
                inputs=[
                    "train_dataset",
                    "train_dataloader",
                    "valid_dataloader",
                    "test_dataloader",
                    "params:model_type",
                    "params:generic_model_params",
                    "params:optim_params",
                    "params:train_params",
                ],
                outputs="trained_model",
                name="train_model",
            ),
        ]
    )

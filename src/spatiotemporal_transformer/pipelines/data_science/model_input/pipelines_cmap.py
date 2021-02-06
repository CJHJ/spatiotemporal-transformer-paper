from kedro.pipeline import node, Pipeline
from spatiotemporal_transformer.pipelines.data_science.model_input.nodes import (
    make_time_series_data,
    prepare_scale_data,
    init_scaler,
    normalize_data,
    denormalize_data,
    split_data,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_cmap", "params:split_parameters"],
                outputs=["train_data", "valid_data", "test_data"],
                name="split_data",
            ),
            node(
                func=make_time_series_data,
                inputs=[
                    "params:data_parameters.sequence_length",
                    "train_data",
                ],
                outputs="train_series_data",
                name="make_time_series_train",
            ),
            node(
                func=make_time_series_data,
                inputs=[
                    "params:data_parameters.sequence_length",
                    "valid_data",
                ],
                outputs="valid_series_data",
                name="make_time_series_valid",
            ),
            node(
                func=make_time_series_data,
                inputs=["params:data_parameters.sequence_length", "test_data"],
                outputs="test_series_data",
                name="make_time_series_test",
            ),
            node(
                func=prepare_scale_data,
                inputs=["train_data", "params:scaler_parameters"],
                outputs="data_for_fitting",
                name="prepare_scale_data",
            ),
            node(
                func=init_scaler,
                inputs=["params:scaler_parameters", "data_for_fitting"],
                outputs="scaler",
                name="initializing_scaler",
            ),
            node(
                func=normalize_data,
                inputs=["scaler", "train_series_data"],
                outputs="norm_train_data",
                name="normalizing_train_data",
            ),
            node(
                func=normalize_data,
                inputs=["scaler", "valid_series_data"],
                outputs="norm_valid_data",
                name="normalizing_valid_data",
            ),
            node(
                func=normalize_data,
                inputs=["scaler", "test_series_data"],
                outputs="norm_test_data",
                name="normalizing_test_data",
            ),
        ]
    )

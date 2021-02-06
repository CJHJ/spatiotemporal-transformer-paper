from kedro.pipeline import node, Pipeline
from spatiotemporal_transformer.pipelines.data_engineering.nodes import (
    preprocess_cmap,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_cmap,
                inputs="raw_cmap",
                outputs="preprocessed_cmap",
                name="preprocessing_cmap",
            )
        ]
    )

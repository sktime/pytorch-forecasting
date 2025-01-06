import pytest
import torch

from pytorch_forecasting import MultiEmbedding


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(embedding_sizes=(10, 10, 10)),
        dict(embedding_sizes=((10, 3), (10, 2), (10, 1))),
        dict(x_categoricals=["x1", "x2", "x3"], embedding_sizes=dict(x1=(10, 10))),
        dict(
            x_categoricals=["x1", "x2", "x3"],
            embedding_sizes=dict(x1=(10, 2), xg1=(10, 3)),
            categorical_groups=dict(xg1=["x2", "x3"]),
        ),
    ],
)
def test_MultiEmbedding(kwargs):
    x = torch.randint(0, 10, size=(4, 3))
    embedding = MultiEmbedding(**kwargs)
    assert embedding.input_size == x.size(
        1
    ), "Input size should be equal to number of features"
    out = embedding(x)
    if isinstance(out, dict):
        assert isinstance(kwargs["embedding_sizes"], dict)
        for name, o in out.items():
            assert (
                o.size(1) == embedding.output_size[name]
            ), "Output size should be equal to number of embedding dimensions"
    elif isinstance(out, torch.Tensor):
        assert isinstance(kwargs["embedding_sizes"], (tuple, list))
        assert (
            out.size(1) == embedding.output_size
        ), "Output size should be equal to number of summed embedding dimensions"
    else:
        raise ValueError(f"Unknown output type {type(out)}")

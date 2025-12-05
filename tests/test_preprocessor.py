import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.constants import REQUIRED_COLUMNS
import pytest

def test_preprocess_pipeline_basic():
    df = pd.DataFrame({
        col: np.random.rand(10) for col in REQUIRED_COLUMNS
    })

    pre = DataPreprocessor()
    processed = pre.preprocess_pipeline(df, mode="train")

    for col in REQUIRED_COLUMNS:
        assert col in processed.columns

    assert processed.shape[1] > len(REQUIRED_COLUMNS)

    assert isinstance(pre.zero_var_columns, list)


def test_validating_required_columns():
    df = pd.DataFrame({
        col: np.random.rand(10) for col in REQUIRED_COLUMNS
    })
    df.drop(REQUIRED_COLUMNS[0], axis=1, inplace=True)

    pre = DataPreprocessor()

    with pytest.raises(ValueError) as exc_info:
        pre.preprocess_pipeline(df, mode="train")
        pre.preprocess_pipeline(df, mode="train")

    assert "Missing required columns:" in str(exc_info.value)

if __name__=='__main__':
    test_preprocess_pipeline_basic()

import pandas as pd
from src.predict_model import Predictor
from src.constants import REQUIRED_COLUMNS

def test_predictor_loads_model_and_preprocessor(monkeypatch):
    # Подменяем загрузку, чтобы predictor не полез в реальные файлы
    class DummyPreprocessor:
        def preprocess_pipeline(self, X, mode, zero_var_columns=None):
            return X

    class DummyModel:
        def predict(self, X):
            return [1] * len(X)

    monkeypatch.setattr("src.predict_model.Predictor.load_preprocessor", lambda self: DummyPreprocessor())
    monkeypatch.setattr("src.predict_model.Predictor.load_model", lambda self: DummyModel())

    pr = Predictor()

    df = pd.DataFrame({col: [0.1] for col in REQUIRED_COLUMNS})

    pred = pr.predict(df)

    assert pred == [1]

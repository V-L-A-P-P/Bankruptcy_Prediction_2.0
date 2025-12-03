from src.data_preprocessing import DataPreprocessor
from src.hyperparameters_tuning import HyperparameterTuner, CatboostHyperparameterTuner
import pandas as pd
from config.config import RAW_DATA_DIR, MODELS_DIR
from src.constants import TARGET_COLUMN
from catboost import CatBoostClassifier
import pickle

class ModelTrainer:
    def __init__(self, preprocessor : DataPreprocessor, hyperparameter_tuner: HyperparameterTuner):
        self.preprocessor = preprocessor
        self.hyperparameter_tuner = hyperparameter_tuner

    def train(self, X, y, model_name='catboost', params=None):
        X = self.preprocessor.preprocess_pipeline(X, mode='train')

        if params is None:
            study = self.hyperparameter_tuner.start_tuning(X, y, n_trials=5)
            if model_name=='catboost':
                model = CatBoostClassifier(**study.best_params)
            else:
                raise NameError(f"Incorrect model name: '{model_name}'")
        else:
            if model_name=='catboost':
                model = CatBoostClassifier(**params, eval_metric='BalancedAccuracy')
            else:
                raise NameError(f"Incorrect model name: '{model_name}'")

        model.fit(X, y, verbose=200)
        self.save_results(model=model,
                          preprocessor=self.preprocessor)
        return model, self.preprocessor

    def save_results(self, model, preprocessor):
        with open(MODELS_DIR + '/model.pkl', "wb") as f:
            pickle.dump(model, f)

        with open(MODELS_DIR + '/preprocessor.pkl', "wb") as f:
            pickle.dump(preprocessor, f)

        print('Model and preprocessor were saved to /models')

if __name__ == "__main__":
    data = pd.read_csv(RAW_DATA_DIR + '/train.csv').iloc[:200]
    trainer = ModelTrainer(DataPreprocessor(), CatboostHyperparameterTuner())
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    model, preprocessor = trainer.train(X, y, model_name='catboost')
    print(model, preprocessor)
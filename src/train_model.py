from src.data_preprocessing import DataPreprocessor
from src.hyperparameters_tuning import HyperparameterTuner, CatboostHyperparameterTuner
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
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

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if params is None:
            study = self.hyperparameter_tuner.start_tuning(X_train, y_train, n_trials=5)
            if model_name=='catboost':
                model = CatBoostClassifier(**study.best_params)
            else:
                raise NameError(f"Incorrect model name: '{model_name}'")
        else:
            if model_name=='catboost':
                model = CatBoostClassifier(**params, eval_metric='BalancedAccuracy')
            else:
                raise NameError(f"Incorrect model name: '{model_name}'")

        model.fit(X_train, y_train, verbose=200)
        calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrator.fit(X_val, y_val)

        self.save_results(model=model,
                          preprocessor=self.preprocessor,
                          calibrated_model=calibrator)
        return model, self.preprocessor, calibrator

    def save_results(self, model, preprocessor, calibrated_model):
        with open(MODELS_DIR + '/model.pkl', "wb") as f:
            pickle.dump(model, f)

        with open(MODELS_DIR + '/preprocessor.pkl', "wb") as f:
            pickle.dump(preprocessor, f)

        with open(MODELS_DIR + '/calibrated_model.pkl', "wb") as f:
            pickle.dump(calibrated_model, f)

        print('Model, calibrated model, and preprocessor were saved to /models')

if __name__ == "__main__":
    data = pd.read_csv(RAW_DATA_DIR + '/train.csv').iloc[:3000]
    trainer = ModelTrainer(DataPreprocessor(), CatboostHyperparameterTuner())
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    model, preprocessor, calibrated_model = trainer.train(X, y, model_name='catboost')
    print(model, preprocessor)
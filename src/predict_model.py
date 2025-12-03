import pandas as pd
from config.config import MODELS_DIR, RAW_DATA_DIR
import pickle

class Predictor:
    def __init__(self):
        self.preprocessor = self.load_preprocessor()
        self.model = self.load_model()

    def load_preprocessor(self):
        try:
            with open(MODELS_DIR + "/preprocessor.pkl", "rb") as f:
                return pickle.load(f)
        except:
            raise Exception("Train model at first!")

    def load_model(self):
        try:
            with open(MODELS_DIR + "/model.pkl", "rb") as f:
                return pickle.load(f)
        except:
            raise Exception("Train model at first!")

    def predict(self, X):
        X = self.preprocessor.preprocess_pipeline(X, mode='eval')
        return self.model.predict(X)

if __name__ == "__main__":
    X = pd.read_csv(RAW_DATA_DIR+'/test.csv')
    model = Predictor()
    print(model.predict(X))
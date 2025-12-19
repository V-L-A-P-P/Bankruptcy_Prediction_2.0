from fastapi import FastAPI, UploadFile, File, HTTPException
from src.predict_model import Predictor
from src.constants import REQUIRED_COLUMNS
import pandas as pd
from pydantic import BaseModel

app = FastAPI(
    title="Bankruptcy Prediction API",
    description="Принимает CSV и возвращает предсказания модели",
    version="1.0.0"
)

# Загружаем модель и препроцессор один раз при старте сервера
predictor = Predictor()

class PredictResponse(BaseModel):
    n_samples: int
    predictions: list[float]

@app.post("/predict_csv", response_model=PredictResponse)
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть формата CSV"
        )

    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Ошибка чтения CSV. Проверьте формат файла."
        )

    print("CSV uploaded:", df.shape)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Отсутствуют обязательные столбцы: {missing}"
        )

    preds = predictor.predict(df)

    # --- КРИТИЧЕСКАЯ ЧАСТЬ ---
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    else:
        preds = list(preds)

    print("Example preds:", preds[:10])

    return PredictResponse(
        n_samples=len(preds),
        predictions=preds
    )

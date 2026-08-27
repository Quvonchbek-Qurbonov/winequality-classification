from fastapi import APIRouter, Query, HTTPException

from app.schemas.schema import Models, DatasetCondition
from app.services.service import train_model
from app.ml_models.ensemble import ordinary_training, hyperparameter_tuning

router = APIRouter(tags=["classification"])


@router.get("/model/train", status_code=200)
def train_model_with_dataset(model_name: Models, dataset_condition: DatasetCondition):
    return train_model(model_name, dataset_condition)


@router.get("dataset/info", status_code=200)
def get_metrics(hyper_tuning: bool, trash_data: bool):
    if hyper_tuning:
        return hyperparameter_tuning(trash_data)
    return ordinary_training(trash_data)
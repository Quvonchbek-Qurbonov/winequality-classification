from fastapi import APIRouter, Query, HTTPException

from app.schemas.schema import Models, DatasetCondition
from app.services.service import train_model
from app.ml_models import bayes, decision_tree, ensemble, knn, logistic, random_forest, svm

router = APIRouter(tags=["classification"])

# Mapping enum values to their respective module implementations
MODEL_MAP = {
    'bayes': bayes,
    'decision_tree': decision_tree,
    'ensemble': ensemble,
    'knn': knn,
    'logistic': logistic,
    'random_forest': random_forest,
    'svm': svm
}


@router.get("/dataset/info", status_code=200)
def get_metrics(
    model_name: str,
    hyper_tuning: bool = False,
    trash_data: bool = False
):
    selected_model = MODEL_MAP.get(model_name)
    if not selected_model:
        raise HTTPException(status_code=404, detail="Model implementation not found")

    if hyper_tuning:
        return selected_model.hyperparameter_tuning(trash_data)
    return selected_model.ordinary_training(trash_data)
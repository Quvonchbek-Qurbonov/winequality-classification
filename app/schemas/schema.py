from typing import List
from pydantic import BaseModel
from enum import Enum

class Models(str, Enum):
    naive_bayes = 'naive_bayes'
    svm = 'svm'
    logistic = 'logistic'
    decision_tree = 'decision_tree'
    ensemble = 'ensemble'
    knn = 'knn'
    random_forest = 'random_forest'

class DatasetCondition(str, Enum):
    unchanged = "unchanged"
    broken = "broken"
    preprocessed = "preprocessed"

class ClassificationRequest(BaseModel):
    testingData: List[int]


class Metrics(BaseModel):
    trainingTime: float|None
    testingTime: float|None
    accuracy: float
    precision: float
    recall: float
    f1: float


class ClassificationResponse(BaseModel):
    model: str
    prediction: int
    metrics: Metrics
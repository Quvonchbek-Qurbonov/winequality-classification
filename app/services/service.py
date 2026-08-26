from typing import List
import timeit
from app.schemas.schema import ClassificationResponse, Metrics, Models, DatasetCondition


def train_model(model_name: Models, dataset_condition: DatasetCondition):

    start_time = timeit.default_timer()
    training_time = timeit.default_timer() - start_time


    return ClassificationResponse(
        model=model_name,
        prediction=1,
        metrics=Metrics(
            trainingTime=1.1,
            testingTime=1.1,
            accuracy=1,
            precision=1,
            recall=1,
            f1=1.1
        )
    )
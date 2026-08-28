import joblib
from datetime import datetime
from pathlib import Path


def save_model(selected_model, model_name):

    model_name = f"{model_name}.{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"

    path = Path('app/static/models') / model_name
    joblib.dump(selected_model, path)

def load_model(model_name):
    path = Path('app/static/models') / model_name
    selected_model = joblib.load(path)
    return selected_model

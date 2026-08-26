import pandas as pd
import numpy as np
import random
from app.core.config import settings


class Data:

    @staticmethod
    def get_data():
        data = pd.read_csv(settings.DATASET)
        X = data.drop['quality', 'Id']
        y = data['quality']
        return X, y

    @staticmethod
    def get_broken_data():
        data = pd.read_csv(settings.DATASET)

        y = data['quality']
        X = data.drop(columns=['quality', 'Id'])

        random.seed(1000)

        for col in X.columns:
            for i in range(len(X)):
                choice = random.randint(0, 30)

                if choice == 1:
                    X.at[i, col] = random.randint(1000, 10000)
                elif choice == 2:
                    X.at[i, col] = np.nan
        return X, y

    @staticmethod
    def get_processed_data():
        data = pd.read_csv(settings.DATASET)
        X = data.drop['quality', 'Id']
        y = data['quality']
        return data
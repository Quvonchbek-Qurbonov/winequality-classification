import pandas as pd
import numpy as np
import random
from app.core.config import settings
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(settings.DATASET)


class Data:

    @staticmethod
    def get_data():
        X = data.drop(columns=['quality', 'Id'])
        y = data['quality']
        return X, y

    @staticmethod
    def get_broken_data():

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
    def preprocess_data(X, y):
        data = pd.concat([X, y], axis=1)
        data.drop_duplicates(keep='first', inplace=True)  # Drop duplicates from dataset
        data.dropna(inplace=True)  # Drop null values from dataset

        X = data.drop(columns=['quality'])
        y = data['quality']

        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()

            lower_bound = mean - 2*std
            upper_bound = mean + 2*std

            mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
            X = X[mask]
            y = y[mask]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y
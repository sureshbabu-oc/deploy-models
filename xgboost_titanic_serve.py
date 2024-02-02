# This script shows you how to predict titanic passengers survivals.
from ray import serve
import os
import logging

from io import BytesIO, StringIO
from starlette.requests import Request
from typing import Dict

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from data_frame_imputer import DataFrameImputer

feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns = ['Sex']

def transform_feature(test_df, train_df):
    # Join the features from train and test together before imputing missing values,
    # in case their distribution is slightly different
    big_X = pd.concat([train_df[feature_columns_to_use], test_df[feature_columns_to_use]])
    big_X_imputed = DataFrameImputer().fit_transform(big_X)
    # XGBoost doesn't (yet) handle categorical features automatically, so we need to change
    # them to columns of integer values.
    le = LabelEncoder()
    for feature in nonnumeric_columns:
        big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])
    # Prepare the inputs for the model
    test_X = big_X_imputed[train_df.shape[0]::].to_numpy()
    return test_X

@serve.deployment
class TitanicModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
        self.model.load_model(os.environ["MODEL_PATH"]+'/model.xgb')
        self.logger = logging.getLogger("ray.serve")
        encoder_file = os.path.abspath('train.csv')
        self.train_df = pd.read_csv(encoder_file, header=0)

    async def __call__(self, starlette_request: Request) -> Dict:
        # Get csv file data.
        data = await starlette_request.body()
        df = pd.read_csv(BytesIO(data))
        self.logger.info("[1/3] Recived csv file: {}".format(df))

        values = transform_feature(df, self.train_df)

        values = self.model.predict(values)
        self.logger.info("[2/3] Predicted values from Model: {}".format(values))

        prediction = ["Dead" if pred == 0 else "Alive" for pred in values]
        self.logger.info("[3/3] Inference done!")

        return {"Predicted Output:": prediction}

deploy = TitanicModel.bind()

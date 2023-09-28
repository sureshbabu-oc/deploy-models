from ray import serve
import os
import logging

from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict

import pandas as pd
from io import StringIO
import pickle
#import joblib
from sklearn import preprocessing

##############################################
features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

def encode_features(df_train, df_test): #label encoding for sklearn algorithms
    df_combined = pd.concat([df_train[features], df_test[features]])
    encoders = {}
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
        encoders[feature] = le
    return df_train, df_test, encoders

def simplify_ages(df): # Binning ages 
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df): # Storing first letter of cabins
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df): # Binning fares
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df): # Keeping title
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df): # Dropping useless features
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df
##############################################

@serve.deployment
class TitanicModel:
    def __init__(self):
        self.model = pickle.load(open(os.environ["MODEL_PATH"]+'/model.pkl','rb'))
        self.logger = logging.getLogger("ray.serve")
        encoder_file = os.path.abspath('encoders.pkl')
        self.encoders = pickle.load(open(encoder_file, "rb"))

    async def __call__(self, starlette_request: Request) -> Dict:
        # Get csv file data.
        data = await starlette_request.body()
        df = pd.read_csv(BytesIO(data))
        self.logger.info("[1/3] Recived csv file: {}".format(df))

        df = transform_features(df)
        for feature in features:
            le = self.encoders[feature]
            df[feature] = le.transform(df[feature])

        values = df.drop(["Survived", "PassengerId"], 1, errors='ignore').values
        values = self.model.predict(values)
        self.logger.info("[2/3] Predicted values from Model: {}".format(values))

        prediction = ["Dead" if pred == 0 else "Alive" for pred in values]
        self.logger.info("[3/3] Inference done!")

        return {"Predicted Output:": prediction}

deploy = TitanicModel.bind()

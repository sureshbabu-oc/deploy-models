# deploy-models
## Deploy a Model & Prediction:
  ## Sklearn:
    ## Deploy:
      - d3x serve create -n <deployment-name> -r mlflow --model <model-name> --model_version 1 --depfilepath sklearn_titanic_serve.deploy
    ## Prediction:
      - python client/sklearn_titanic_client.py <profile-name> <deployment-name> client/test_file/test.csv
 ## Tensorflow:
    Deploy:
      - d3x serve create -n <deployment-name> -r mlflow --model <model-name> --model_version 1 --depfilepath tensorflow_mnist_serve.deploy
    Prediction:
      - python client/tensorflow_mnist_client.py <profile-name> <deployment-name> client/images/3.png
 ## Xgboost:
    Deploy:
      - d3x serve create -n <deployment-name> -r mlflow --model <model-name> --model_version 1 --depfilepath xgboost_titanic_serve.deploy
    Prediction:
      - python client/xgboost_titanic_client.py <profile-name> <deployment-name> client/test_file/test.csv
  ## Pytorch:
    Deploy:
      - d3x serve create -n <deployment-name> -r mlflow --model <model-name> --model_version 1 --depfilepath pytorch_mnist_serve.deploy
    Prediction:
      - python client/tensorflow_mnist_client.py <profile-name> <deployment-name> client/images/3.png
  ## Custom-model:
    Deploy:
      - d3x serve create -n <deployment-name> -r mlflow --model <model-name> --model_version 1 --depfilepath custom_mnist_serve.deploy
    Prediction:
      - python client/tensorflow_mnist_client.py <profile-name> <deployment-name> client/images/3.png
  

import mlflow.pytorch
from ray import serve
import os
import logging

from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict

import pandas as pd
import numpy as np
from io import StringIO
import pickle
import cv2
import base64

img_w = 28
img_h = 28

# def b64_filewriter(filename, content):
#     string = content.encode('utf8')
#     b64_decode = base64.decodebytes(string)
#     fp = open(filename, "wb")
#     fp.write(b64_decode)
#     fp.close()

@serve.deployment
class ImageModel:
    def __init__(self):
        self.model = mlflow.pytorch.load_model(os.environ["MODEL_PATH"])
        self.logger = logging.getLogger("ray.serve")

    async def __call__(self, starlette_request: Request) -> Dict:
        # Get Image in bytes formats.
        image_payload_bytes = await starlette_request.body()
        self.logger.info("[1/2] Image payload recieved: {}".format(image_payload_bytes))

        #TODO Need to fix prediction Error.
        #image_payload_string = image_payload_bytes.decode("utf-8")
        with open("image.png", "wb") as fh:
            fh.write(base64.decodebytes(image_payload_bytes))
        img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28,28))
        img = img.reshape(1,1,img.shape[0],img.shape[1])
        img = img.astype('float32')

        # Perform prediction
        #prediction = self.model.predict(x)
        prediction = self.model(img)
        prediction = np.argmax(prediction[0])
        self.logger.info("[2/2] Predicted image : {}".format(prediction))
        return {"Predicted digit:": prediction}

deploy = ImageModel.bind()

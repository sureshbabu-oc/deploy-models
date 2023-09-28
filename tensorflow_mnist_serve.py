from ray import serve
import os
import logging

from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict

import tensorflow as tf
import numpy as np
#import cv2

img_w = 28
img_h = 28

@serve.deployment
class ImageModel:
    def __init__(self):
        self.model = tf.keras.models.load_model(os.environ["MODEL_PATH"]+'/data/model')
        self.logger = logging.getLogger("ray.serve")

    async def __call__(self, starlette_request: Request) -> Dict:
        # Get Image in bytes formats.
        image_payload_bytes = await starlette_request.body()
        pil_image = Image.open(BytesIO(image_payload_bytes))
        pil_image = pil_image.resize((img_w, img_h))
        self.logger.info("Parsed image data: {}".format(pil_image))

        x = np.array(pil_image, dtype=np.float64)
        #x = cv2.resize(x, (img_w, img_h))
        x = x.reshape(1,img_w, img_h,1)

        # Perform prediction
        prediction = self.model.predict(x)
        prediction = np.argmax(prediction[0])
        self.logger.info("Predicted image : {}".format(prediction))
        return {"Predicted digit:": prediction}

deploy = ImageModel.bind()

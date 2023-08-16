from pydantic import BaseModel
from fastapi import FastAPI

import numpy as np
import tensorflow as tf
from tensorflow import keras
import base64

SAVED_MODEL_PATH = "./model/model.h5"

CLASSIFY_MODEL = keras.models.load_model(SAVED_MODEL_PATH)

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 512 , 384 , 3

app = FastAPI()

class RequestInput(BaseModel):
    input: str

#get 
# @app.get("/")
# async def index():
#     return {"Message": ["Hello World"]}

@app.post("/predict")
async def predict(request: RequestInput):
    print(request.input)
    request_input = BaseModel(
        target_datatype=np.float32, 
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        image_channel=IMAGE_CHANNEL,
    )(request.input)

    prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)

    return {"prediction": prediction.tolist()}
#json
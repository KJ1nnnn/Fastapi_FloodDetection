from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.encoders import jsonable_encoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
import base64
from PIL import Image
from io import BytesIO

SAVED_MODEL_PATH = "./model/model.h5"

app = FastAPI()

CLASSIFY_MODEL = keras.models.load_model(SAVED_MODEL_PATH)
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 512, 384, 3

class RequestInput(BaseModel):
    input: dict

def decode_base64_image(encoded_image):
    image_data = base64.b64decode(encoded_image)
    return Image.open(BytesIO(image_data))

def preprocess_image(image):
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.array(image)
    image = image.astype(np.float32) / 255.0  # 픽셀 값 정규화
    return image

@app.post("/")
async def index():
    return {
        "Request Message": "Hello, World!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        preprocessed_image = preprocess_image(image)
        
        prediction = CLASSIFY_MODEL.predict(np.expand_dims(preprocessed_image, axis=0))
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        
        class_mapping = {0: "flooding", 1: "no flooding"}
        predicted_label = class_mapping.get(predicted_class, "unknown")
        
        return {"prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
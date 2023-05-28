import io
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle
import os

app = FastAPI()
file_path = os.path.join(os.getcwd(), "final_model.h5")
model = tf.keras.models.load_model(file_path)
model_name = "Recycle classification"
version = "v1.0.0"

def preprocess_image(image):
    resized_image = tf.image.resize(image, [150, 150])
    # Normalize the image
    normalized_image = resized_image / 255.0
    # Expand dimensions to create a batch of size 1
    preprocessed_image = tf.expand_dims(normalized_image, axis=0)

    return preprocessed_image

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Read and process the uploaded image
    image = await file.read()
    img = Image.open(io.BytesIO(image))
    # Convert PIL image to TensorFlow Tensor
    img_tensor = tf.convert_to_tensor(np.array(img))
    # Preprocess the image
    processed_img = preprocess_image(img_tensor)

    # Prediction and return the result
    probability = model.predict(processed_img)
    result = "Recyclable" if probability.round() == 1 else "Not Recyclable"
    return {
        "result": result,
        "probability": str(probability)
        }

@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

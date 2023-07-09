from typing import Union
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import joblib 
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the trained model
model = joblib.load("obeseLevelPred.joblib")


@app.get("/")
def read_root():
    return {"Hello": "Goloo"}


@app.get("/ping")
def read_root():
    return {"ans":"Tenga"}


@app.get("/ting")
def read_root():
    return {"ans":"manga"}

# Define the input data model
class InputData(BaseModel):
    data: list[float]

# Define the prediction endpoint
# @app.post("/predict")
# async def predict(input_data: InputData):
#     # Convert the input data to a numpy array
#     X = np.array([input_data.data])
    
#     # Make the prediction
#     y_pred = model.predict(X)
    
#     # Return the prediction as a dictionary
#     return {"prediction": y_pred[0]}


#app.add_middleware(CORSMiddleware, allow_origins=['*'])

class Data(BaseModel):
    data: list[float]

@app.post('/predict')
def predict(data: Data):
    print("Data",[data.data])
    model = joblib.load("obeseLevelPred.joblib")
    prediction = model.predict([data.data])
    return {'result': str(prediction[0])}










# @app.post('/predict')
# async def predict(array: List[float]):
#     print([array])
#     clf = joblib.load("obeseLevelPred.joblib") # Replace with the path to your saved joblib model
#     prediction = clf.predict([array])[0]
#     return {"prediction": prediction}



# @app.post("/predict")
# def predict(data: dict):
#     testrun = [0,21.0,1.62,64.0,1,3.0,0,2.0,0.0,3]
#     # Convert input data to a format that the model can understand

#     #X = [[data["feature1"], data["feature2"], data["feature3"],data["feature4"], data["feature5"], data["feature6"],data["feature7"], data["feature8"], data["feature9"], data["feature10"]]]
#     #X = [[data[0], data[21.0], data[1.62],data[64.0], data[1], data[3.0],data[0], data[2.0], data[0.0],data[3]]]
#     data = {"feature1": 0, "feature2": 21.0, "feature3": 1.62 , "feature4": 64.0, "feature5": 1, "feature6": 3.0, "feature7": 0 , "feature8": 2.0, "feature9": 0.0, "feature10": 3}
#     # Make predictions using the loaded model
#     y_pred = model.predict(data)
#     print (y_pred)
#     # Return the predictions
#     return {"prediction": y_pred[0]}
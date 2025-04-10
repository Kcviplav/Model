from fastapi import FastAPI
from pydantic import BaseModel
#from sklearn.preprocessing import OneHotEncoder
#import numpy as np 
import pandas as pd
import joblib


# df = pd.read_csv("../bengaluru_house_pprices.csv")
# location_dummies = pd.get_dummies(df['location'], prefix = 'location')
# dummy_columns = location_dummies.columns.tolist()
# joblib.dump(dummy_columns,'dummy_columns.pkl')
# print("dummy columns saved",dummy_columns[:5],"...")


app = FastAPI()
model = joblib.load("bangalore_homes_prices_model.pickle")
dummy_columns = joblib.load("dummy_columns.pkl")

class InputData(BaseModel):
    place: str
    bathroom: int
    total_sqft: int 
    rooms: int 
    
@app.get("/")
def read_root():
    return{"message":"Welcome to the website !!!"}
@app.post("/predict")
def predict(data :InputData):
    # creating dummy from input 
    place_df = pd.DataFrame({'location': [data.place]})
    place_dummies = pd.get_dummies(place_df['location'],prefix= 'location')
    
    # aligning dummies from input 
    for col in dummy_columns:
        if col not in place_dummies:
            place_dummies[col] = 0
    
    place_dummies = place_dummies[dummy_columns]
    
    
    features=  pd.DataFrame({
        'bathroom': [data.bathroom],
        'Sq_foot': [data.total_sqft],
        'total_rooms': [data.rooms]
    })
    
    final_input = pd.concat([features,place_dummies],axis =1 )
    
    prediction = model.predict(final_input)
    return{"prediction": prediction[0]}
    
   
    
    



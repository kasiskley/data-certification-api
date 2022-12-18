
from fastapi import FastAPI
import joblib
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get('/predict')
def predict(acousticness:float, danceability:float, duration_ms:int, energy:float, \
            explicit:int, id:str, instrumentalness:float, key:int, liveness:float, \
            loudness:float, mode:int, name:str, release_date:str, speechiness:float, \
            tempo:float, valence:float, artist:str):
    try:
        # reconstitution des features pour la prédiction
        # st = StandardScaler()
        data = {'acousticness':[acousticness], 'danceability':[danceability], 'duration_ms':[duration_ms], 'energy':[energy], \
                'explicit':[explicit], 'id':[id], 'instrumentalness':[instrumentalness], 'key':[key], 'liveness':[liveness], \
                'loudness':[loudness], 'mode':[mode], 'name':[name], 'release_date':[release_date], 'speechiness':[speechiness], \
                'tempo':[tempo], 'valence':[valence], 'artist':[artist]}
        data_df = pd.DataFrame.from_dict(data)
        # récupération du model enregistré pour prédiction
        mdl = joblib.load('model.joblib')
        res =  mdl.predict(data_df)
        return {'artist' : artist,
                'name': name,
                'popularity':res[0]}
    except:
        return {"error": str(sys.exc_info()[1])}

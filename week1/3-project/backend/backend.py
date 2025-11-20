from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from typing import List
import traceback
import os
port = int(os.environ.get("PORT", 8000))

# Modèle Pydantic (types basés sur ton info())
class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: float
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str

app = FastAPI()

# Chargement des modèles
try:
    model = joblib.load("modele/modele.joblib")
    ohe = joblib.load("modele/ohe.joblib")
    print("Modèles chargés avec succès !")
    print("Colonnes atte3ndues par OHE :", ohe.feature_names_in_)  
except Exception as e:
    print(f"Erreur de chargement : {str(e)}")


categorical_cols = [
    'customerID', 'gender', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity'
]

@app.post("/predict")
def predict(donnees: List[Customer]):
    try:
        # Convertir en DataFrame
        data_list = [c.dict() for c in donnees]
        df = pd.DataFrame(data_list)
        print("DataFrame reçu :", df.head())
        
        # Sauvegarder customerID pour le retour
        customer_ids = df["customerID"].copy()
        
        # Sélectionner SEULEMENT les colonnes catégorielles (comme dans entraînement)
        features = df[categorical_cols]
        print("Colonnes sélectionnées pour OHE :", features.columns.tolist())
        print("Shape des features :", features.shape)
        
        # Transformer (ohe attend des strings, donc OK)
        encoded_sparse = ohe.transform(features)
        encoded_df = pd.DataFrame(
            encoded_sparse.toarray(),
            columns=ohe.get_feature_names_out()
        )
        print("Shape après encoding :", encoded_df.shape)
        
        # Prédiction
        pred = model.predict(encoded_df)
        pred_labels = np.where(pred == 0, "Yes", "No")  # Comme dans ton code
        
        # Résultat
        result = pd.DataFrame({
            "customerID": customer_ids+'0',
            "Churn": pred_labels
        })
        
        return result.to_dict(orient="records")
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Erreur dans /predict : {error_trace}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}\nTrace : {error_trace}")

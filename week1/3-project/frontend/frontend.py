import streamlit as st
import requests
import pandas as pd

st.title("Churn prediction mod")
st.markdown("Pour prédire le comportement de vos clients, chargez le fichier CSV contenant les caractéristiques suivantes sur vos clients:")
st.markdown("""**['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity']**""")

uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if uploaded_file is not None:
    donnees = pd.read_csv(uploaded_file)
    st.write("Données chargées avec succès")
    st.markdown("Aperçu des données")
    st.dataframe(donnees)
    
    if st.button("Prédire"):
        # Envoyer les données en JSON (list de dicts)
        response = requests.post("https://mlops-vx6k.onrender.com/predict", json=donnees.to_dict(orient='records'))
        
        if response.status_code == 200:
            predictions = response.json()
            st.write("Les prédictions sont :")
            st.dataframe(pd.DataFrame(predictions))  # Afficher en table pour plus de clarté
        else:
            st.error(f"Erreur lors de la prédiction : {response.text}")

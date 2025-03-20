import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import json
from PIL import Image

# Définir les URL de l'API en fonction du choix
API_URL = "http://34.207.61.70:80"

st.set_page_config(page_title="Dashboard_Prediction",page_icon="📈",layout="centered")
st.title("🔍 Service client - Solvabilité")

image = Image.open("data/pret_a_financer_img.png")
st.image(image,caption="Prêt à dépenser",width=550)

# Vérifier si un client existe
st.sidebar.header("📌 Choisir un client")
client_id = st.sidebar.number_input("L'identifiant du client", min_value=1, step=1, format="%d")

# Récupérer les infos d’un client
st.header("📄 Informations client")
if st.button("🔍 Obtenir les infos du client"):
    response = requests.get(f"{API_URL}/client_info/{client_id}")
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Client non trouvé ou erreur API.")

# Prédiction
st.header("🔮 Prédire la solvabilité du client")
if st.button("📊 Obtenir la prédiction"):
    response = requests.post(f"{API_URL}/predict", json={"client_id": client_id})
    if response.status_code == 200:
        result = response.json()
        predicted_proba_default = result['probabilité']
        best_threshold = 0.54
        client_status = "Client non solvable" if predicted_proba_default >= best_threshold else "Client solvable"
        st.success(f"Probabilité de défaut de paiement : {predicted_proba_default:.2f}")
        st.info(f"Statut : {client_status}")
    else:
        st.error("Client non trouvé ou erreur API.")

# Affichage des graphes SHAP
st.header("📈 Importance des features")
# Récupérer les 15 features les plus importantes globalement
if st.button("🌍 Importance globale"):
    response_global = requests.get(f"{API_URL}/feature_importance_global")
    if response_global.status_code == 200:
        df_global = pd.DataFrame(response_global.json().items(), columns=["Feature", "Importance Globale"])
        top_15_features_global = df_global.sort_values(by="Importance Globale", ascending=False).head(15)
        st.write("Prédiction de la solvabilité - Features les plus influents globalement")
        st.bar_chart(top_15_features_global.set_index("Feature"))
    else:
        st.error("Erreur lors de la récupération des données.")

# Récupérer les 15 features les plus importantes localement
if st.button("👤🏡 Importance locale (Client)"):
    response_local = requests.get(f"{API_URL}/feature_importance_local/{client_id}")
    response_global = requests.get(f"{API_URL}/feature_importance_global")
    
    if response_local.status_code == 200 and response_global.status_code == 200:
        df_local = pd.DataFrame(response_local.json().items(), columns=["Feature", "Importance Locale"])
        top_15_features_local = df_local.sort_values(by="Importance Locale", ascending=False).head(15)

        st.write("Prédiction de la solvabilité - Features les plus influents localement")
        st.bar_chart(top_15_features_local.set_index("Feature"))

        df_global = pd.DataFrame(response_global.json().items(), columns=["Feature", "Importance Globale"])
        top_15_features_global = df_global.sort_values(by="Importance Globale", ascending=False).head(15)
        
        # Fusionner les deux DataFrames pour comparer
        df_comparison = top_15_features_local.merge(top_15_features_global, on="Feature", how="left").fillna(0)
        df_comparison = df_comparison.sort_values(by="Importance Locale", ascending=False)
        
        st.write("Features influents localement vs features influents globalement")
        st.dataframe(df_comparison)
        st.bar_chart(df_comparison.set_index("Feature"))
    else:
        st.error("Client non trouvé ou erreur API.")

st.header("📊📉 Visualisation des caractéristiques client")
# Récupérer les 15 features les plus importantes globalement
response_global = requests.get(f"{API_URL}/feature_importance_global")
if response_global.status_code == 200:
    df_global = pd.DataFrame(response_global.json().items(), columns=["Feature", "Importance Globale"])
    top_15_features_global_list = df_global.sort_values(by="Importance Globale", ascending=False).head(15)["Feature"].tolist()

    # Sélection de la première caractéristique
    feature_x = st.selectbox("🚀 Sélectionner une feature à visualiser", top_15_features_global_list)

    if st.button("📊 Afficher la visualisation"):
        # Récupérer les données pour la première feature
        response_feature_x = requests.get(f"{API_URL}/feature_distribution/{feature_x}")
        if response_feature_x.status_code == 200:
            # La réponse de l'API est un dictionnaire, nous devons extraire la liste des valeurs
            feature_x_data = response_feature_x.json().get(feature_x, [])
            
            fig, ax = plt.subplots()
            # Affichage de l'histogramme pour la feature_x
            sns.histplot(feature_x_data, bins=30, kde=True, ax=ax, color="blue", label=feature_x)
            st.pyplot(fig)
        else:
            st.error(f"Erreur lors de la récupération des données pour {feature_x}")

    # Liste des features restantes pour la comparaison
    remaining_features = [feat for feat in top_15_features_global_list if feat != feature_x]
    # Sélection de la seuxième caractéristique
    feature_y = st.selectbox("🚀 Sélectionner une nouvelle feature", remaining_features)
    if feature_y:
        if st.button("📊 Analyse Bi-Variée"):
            # Vérification de la présence de feature_y et récupération de ses données
            response_feature_x = requests.get(f"{API_URL}/feature_distribution/{feature_x}")
            response_feature_y = requests.get(f"{API_URL}/feature_distribution/{feature_y}")
            if response_feature_x.status_code == 200 and response_feature_y.status_code == 200:
                # Extraire les données de feature_y et de feature_x
                feature_x_data = response_feature_x.json().get(feature_x, [])
                feature_y_data = response_feature_y.json().get(feature_y, [])               
                fig, ax = plt.subplots()
                #Affichage du scatter plot si les données existent
                sns.scatterplot(x=feature_x_data, y=feature_y_data, ax=ax)
                st.pyplot(fig)

            else:
                st.error(f"Erreur lors de la récupération des données pour {feature_y}")
else:
    st.error("Erreur lors de la récupération des features importantes.")


# Mettre à jour les infos d’un client
st.header("📝 Mise à jour des informations du client")
updated_data = st.text_area("Données mises à jour (format JSON)")
if st.button("✅ Mettre à jour"):
    try:
        data = json.loads(updated_data)
        response = requests.put(f"{API_URL}/client_info/{client_id}", json=data)
        if response.status_code == 200:
            st.success("Mise à jour réussie !")
            st.json(response.json())
        else:
            st.error("Erreur lors de la mise à jour.")
    except json.JSONDecodeError:
        st.error("Format JSON invalide.")

# Ajouter un nouveau client
st.header("➕ Ajouter un nouveau client")
new_client_data = st.text_area("Données du nouveau client (format JSON)")
if st.button("🚀 Ajouter"):
    try:
        data = json.loads(new_client_data)
        response = requests.post(f"{API_URL}/client_info", json=data)
        if response.status_code == 200:
            st.success("Client ajouté avec succès !")
            st.json(response.json())
        else:
            st.error("Erreur lors de l'ajout du client.")
    except json.JSONDecodeError:
        st.error("Format JSON invalide.")

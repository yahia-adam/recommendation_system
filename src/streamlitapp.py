import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import requests
from io import BytesIO
import zipfile

from myConst import DATASET_NAME, MODEL_NAME

# Configuration de la page
st.set_page_config(page_title="Recommandation d'Événements",
                   page_icon="🎭",
                   layout="wide",
                   initial_sidebar_state="expanded")

def unzip_dataset():
    zip_file = "datasets.zip"
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(".")
        print(f"✅ Extraction réussie dans '{"."}' !")
    except FileNotFoundError:
        print("❌ Erreur : Le fichier ZIP n'existe pas.")
    except zipfile.BadZipFile:
        print("❌ Erreur : Le fichier n'est pas un ZIP valide.")
unzip_dataset()

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge les données originales et le dictionnaire de similarité"""
    try:
        # Charger le dataset original
        df = pd.read_csv(DATASET_NAME)

        # Charger la matrice de similarité
        with open(MODEL_NAME, 'rb') as f:
            similarity_dict = pickle.load(f)

        return df, similarity_dict
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None

# Fonction pour charger une image depuis une URL
def load_image_from_url(url):
    """Charge une image depuis une URL avec gestion d'erreurs"""
    try:
        if pd.isna(url) or not url:
            return None
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.warning(f"Impossible de charger l'image depuis l'URL: {url}")
        return None


# Fonction pour afficher les détails d'un événement
def display_event(event_row, col):
    """Affiche les détails d'un événement dans une colonne"""
    try:
        # Extraire les informations de l'événement
        event_name = event_row['EVENT_NAME']
        event_id = event_row['EVENT_ID']

        # Informations de base
        with col:
            st.subheader(event_name)
            st.caption(f"ID: {event_id}")

            # Afficher l'image si disponible
            if 'EVENT_IMAGE_URL' in event_row and not pd.isna(event_row['EVENT_IMAGE_URL']):
                img = load_image_from_url(event_row['EVENT_IMAGE_URL'])
                if img:
                    st.image(img, use_container_width=True)

            # Informations sur le lieu
            if 'VENUE_NAME' in event_row and not pd.isna(event_row['VENUE_NAME']):
                st.markdown(f"**Lieu**: {event_row['VENUE_NAME']}")

            if 'VENUE_CITY' in event_row and not pd.isna(event_row['VENUE_CITY']):
                city = event_row['VENUE_CITY']
                if 'VENUE_STATE_CODE' in event_row and not pd.isna(event_row['VENUE_STATE_CODE']):
                    city += f", {event_row['VENUE_STATE_CODE']}"
                st.markdown(f"**Ville**: {city}")

            # Date et heure
            if 'EVENT_START_LOCAL_DATE' in event_row and not pd.isna(event_row['EVENT_START_LOCAL_DATE']):
                date_str = event_row['EVENT_START_LOCAL_DATE']
                if 'EVENT_START_LOCAL_TIME' in event_row and not pd.isna(event_row['EVENT_START_LOCAL_TIME']):
                    date_str += f" à {event_row['EVENT_START_LOCAL_TIME']}"
                st.markdown(f"**Date**: {date_str}")

            # Classification
            if 'CLASSIFICATION_SEGMENT' in event_row and not pd.isna(event_row['CLASSIFICATION_SEGMENT']):
                segment = event_row['CLASSIFICATION_SEGMENT']
                if 'CLASSIFICATION_GENRE' in event_row and not pd.isna(event_row['CLASSIFICATION_GENRE']):
                    segment += f" - {event_row['CLASSIFICATION_GENRE']}"
                st.markdown(f"**Catégorie**: {segment}")

            # Lien vers l'événement
            if 'PRIMARY_EVENT_URL' in event_row and not pd.isna(event_row['PRIMARY_EVENT_URL']):
                st.markdown(f"[Voir plus de détails]({event_row['PRIMARY_EVENT_URL']})")

            # Séparateur
            st.markdown("---")

            # Description (sous un expander)
            if 'EVENT_INFO' in event_row and not pd.isna(event_row['EVENT_INFO']):
                with st.expander("Voir la description"):
                    st.write(event_row['EVENT_INFO'])

    except Exception as e:
        col.error(f"Erreur lors de l'affichage de l'événement: {e}")


# Titre principal de l'application
st.title("🎭 Système de Recommandation d'Événements")
st.markdown("Découvrez des événements similaires à ceux qui vous intéressent !")

# Charger les données
df, similarity_dict = load_data()

if df is not None and similarity_dict is not None:
    # Nettoyer un peu le dataframe pour faciliter la recherche
    if 'EVENT_NAME' in df.columns:
        df['EVENT_NAME'] = df['EVENT_NAME'].fillna("Événement sans nom")

    # Filtrer les événements qui ont des recommandations
    events_with_recommendations = list(similarity_dict.keys())
    filtered_df = df[df['EVENT_ID'].isin(events_with_recommendations)]

    # Créer un widget de recherche
    st.sidebar.header("Rechercher un événement")

    # Options de filtrage
    st.sidebar.subheader("Filtres")

    # Filtre par catégorie
    if 'CLASSIFICATION_SEGMENT' in df.columns:
        categories = ["Tous"] + sorted(df['CLASSIFICATION_SEGMENT'].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Catégorie", categories)

        if selected_category != "Tous":
            filtered_df = filtered_df[filtered_df['CLASSIFICATION_SEGMENT'] == selected_category]

    # Filtre par ville
    if 'VENUE_CITY' in df.columns:
        cities = ["Toutes"] + sorted(df['VENUE_CITY'].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox("Ville", cities)

        if selected_city != "Toutes":
            filtered_df = filtered_df[filtered_df['VENUE_CITY'] == selected_city]

    # Recherche par nom
    search_term = st.sidebar.text_input("Rechercher par nom", "")
    if search_term:
        filtered_df = filtered_df[filtered_df['EVENT_NAME'].str.contains(search_term, case=False, na=False)]

    # Afficher le nombre d'événements trouvés
    st.sidebar.info(f"{len(filtered_df)} événements trouvés")

    # Liste des événements (limité à 50 pour éviter de surcharger l'interface)
    event_list = filtered_df.head(50)

    if len(event_list) > 0:
        # Créer un selectbox avec les événements
        event_options = event_list['EVENT_NAME'].tolist()
        selected_event_name = st.selectbox("Choisissez un événement", event_options)

        # Trouver l'événement sélectionné
        selected_event = event_list[event_list['EVENT_NAME'] == selected_event_name].iloc[0]
        selected_event_id = selected_event['EVENT_ID']

        # Afficher l'événement sélectionné
        st.header("Événement sélectionné")
        display_event(selected_event, st.container())

        # Obtenir les événements similaires
        if selected_event_id in similarity_dict:
            similar_event_ids = [item[0] for item in similarity_dict[selected_event_id]]
            similar_events = df[df['EVENT_ID'].isin(similar_event_ids)]

            # Réordonner les événements similaires dans le même ordre que la liste de similarité
            if not similar_events.empty:
                ordered_similar_events = pd.DataFrame()
                for similar_id, _ in similarity_dict[selected_event_id]:
                    event = similar_events[similar_events['EVENT_ID'] == similar_id]
                    if not event.empty:
                        ordered_similar_events = pd.concat([ordered_similar_events, event])

                # Afficher les événements similaires
                st.header("Événements similaires recommandés")

                # Créer une grille de 2 colonnes pour afficher les événements
                cols = st.columns(2)
                for i, (_, row) in enumerate(ordered_similar_events.iterrows()):
                    col_idx = i % 2  # Alterner entre les colonnes
                    display_event(row, cols[col_idx])

                    # Limiter à 10 recommandations
                    if i >= 9:
                        break
            else:
                st.info("Aucun événement similaire trouvé.")
        else:
            st.warning("Aucune recommandation disponible pour cet événement.")
    else:
        st.warning("Aucun événement ne correspond aux critères de recherche.")
else:
    st.error("Impossible de charger les données. Veuillez vérifier les fichiers de données.")

# Ajouter des informations dans le pied de page
st.sidebar.markdown("---")
st.sidebar.caption("Système de recommandation basé sur la similarité des événements.")
st.sidebar.caption("Développé avec Streamlit et scikit-learn.")
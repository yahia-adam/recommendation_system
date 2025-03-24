import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

from src.myConst import MODEL_NAME, DATASET_NAME


def preprocess_events_data(df):
    """
    Pipeline de prétraitement des données d'événements

    Args:
        df (pandas.DataFrame): Le DataFrame brut

    Returns:
        pandas.DataFrame: Le DataFrame prétraité
    """
    print("Début du prétraitement...")
    print(f"Dimensions initiales: {df.shape}")

    # Créer une copie pour éviter de modifier l'original
    processed_df = df.copy()

    # === 1. NETTOYAGE DES DONNÉES ===

    # Supprimer les colonnes vides ou peu utiles
    empty_cols = [
        'EVENT_NOTES', 'MIN_PRICE', 'MAX_PRICE', 'CURRENCY',
        'RESALE_EVENT_URL', 'PRESALE_NAME', 'PRESALE_DATETIME_RANGE',
        'LEGACY_VENUE_ID', 'PRESALE_NAME.1', 'PRESALE_DESCRIPTION',
        'PRESALE_START_DATETIME', 'PRESALE_END_DATETIME',
        'ACCESSIBLE_SEATING_DETAIL', 'ADA_PHONE', 'ADA_CUSTOM_COPY',
        'ADA_HOURS', 'ACCESSIBILITY_INFO', 'PLEASE_NOTE',
        'IMPORTANT_INFORMATION', 'EVENT_END_LOCAL_DATE',
        'MIN_PRICE_WITH_FEES', 'MAX_PRICE_WITH_FEES',
        'EVENT_END_DATETIME'
    ]

    low_info_cols = [
        'VENUE_COUNTRY_CODE', 'VENUE_TIMEZONE',
        'CLASSIFICATION_TYPE', 'CLASSIFICATION_SUB_TYPE',
        'CLASSIFICATION_TYPE_ID', 'CLASSIFICATION_SUB_TYPE_ID'
    ]

    redundant_cols = [
        'LEGACY_EVENT_ID'
    ]

    all_cols_to_remove = empty_cols + low_info_cols + redundant_cols
    existing_cols = [col for col in all_cols_to_remove if col in processed_df.columns]
    processed_df = processed_df.drop(columns=existing_cols)
    print(f"Colonnes supprimées: {len(existing_cols)}")

    # === 2. GESTION DES VALEURS MANQUANTES ET CARACTÉRISTIQUES SIMPLES ===

    # Définir des valeurs par défaut
    DEFAULT_LAT = 48.8566
    DEFAULT_LON = 2.3522

    # Créer des caractéristiques simples

    # Pour attractions
    if 'ATTRACTION_NAME' in processed_df.columns:
        processed_df['HAS_ATTRACTION'] = processed_df['ATTRACTION_NAME'].notna().astype(int)
    else:
        processed_df['HAS_ATTRACTION'] = 0

    # Pour EVENT_INFO
    if 'EVENT_INFO' in processed_df.columns:
        processed_df['EVENT_INFO'] = processed_df['EVENT_INFO'].fillna("Information non disponible")
        processed_df['EVENT_INFO_LENGTH'] = processed_df['EVENT_INFO'].str.len()
    else:
        processed_df['EVENT_INFO_LENGTH'] = 0

    # Pour EVENT_NAME
    if 'EVENT_NAME' in processed_df.columns:
        processed_df['EVENT_NAME_LENGTH'] = processed_df['EVENT_NAME'].fillna('').str.len()
    else:
        processed_df['EVENT_NAME_LENGTH'] = 0

    # Coordonnées géographiques
    if 'VENUE_LATITUDE' in processed_df.columns and 'VENUE_LONGITUDE' in processed_df.columns:
        processed_df['VENUE_LATITUDE'] = processed_df['VENUE_LATITUDE'].fillna(DEFAULT_LAT)
        processed_df['VENUE_LONGITUDE'] = processed_df['VENUE_LONGITUDE'].fillna(DEFAULT_LON)
    else:
        processed_df['VENUE_LATITUDE'] = DEFAULT_LAT
        processed_df['VENUE_LONGITUDE'] = DEFAULT_LON

    # === 3. CARACTÉRISTIQUES TEMPORELLES ===

    try:
        # Convertir les dates
        for col in ['EVENT_START_DATETIME', 'ONSALE_START_DATETIME', 'ONSALE_END_DATETIME', 'EVENT_START_LOCAL_DATE']:
            if col in processed_df.columns:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')

        # Extraire caractéristiques de date
        if 'EVENT_START_LOCAL_DATE' in processed_df.columns:
            processed_df['EVENT_WEEKDAY'] = processed_df['EVENT_START_LOCAL_DATE'].dt.dayofweek.fillna(0)
            processed_df['EVENT_MONTH'] = processed_df['EVENT_START_LOCAL_DATE'].dt.month.fillna(1)
            processed_df['EVENT_YEAR'] = processed_df['EVENT_START_LOCAL_DATE'].dt.year.fillna(2025)
        else:
            processed_df['EVENT_WEEKDAY'] = 0
            processed_df['EVENT_MONTH'] = 1
            processed_df['EVENT_YEAR'] = 2025

        # Convertir l'heure
        if 'EVENT_START_LOCAL_TIME' in processed_df.columns:
            def safe_time_to_decimal(time_str):
                if pd.isna(time_str):
                    return 12.0
                try:
                    h, m = map(int, time_str.split(':'))
                    return float(h) + float(m) / 60
                except:
                    return 12.0

            processed_df['EVENT_TIME_DECIMAL'] = processed_df['EVENT_START_LOCAL_TIME'].apply(safe_time_to_decimal)
        else:
            processed_df['EVENT_TIME_DECIMAL'] = 12.0

    except Exception as e:
        print(f"Erreur lors du traitement temporel: {e}")
        processed_df['EVENT_WEEKDAY'] = 0
        processed_df['EVENT_MONTH'] = 1
        processed_df['EVENT_YEAR'] = 2025
        processed_df['EVENT_TIME_DECIMAL'] = 12.0

    # === 4. CARACTÉRISTIQUES CATÉGORIELLES SIMPLES ===

    try:
        # Pour EVENT_STATUS
        if 'EVENT_STATUS' in processed_df.columns:
            status_values = ['onsale', 'offsale', 'cancelled', 'rescheduled']
            for status in status_values:
                processed_df[f'STATUS_{status}'] = (processed_df['EVENT_STATUS'] == status).astype(int)

        # Pour CLASSIFICATION_SEGMENT
        if 'CLASSIFICATION_SEGMENT' in processed_df.columns:
            segment_values = ['Arts & Theatre', 'Music', 'Sports', 'Miscellaneous', 'Film']
            for segment in segment_values:
                processed_df[f'SEGMENT_{segment}'] = (processed_df['CLASSIFICATION_SEGMENT'] == segment).astype(int)

        # Pour CLASSIFICATION_GENRE (limité aux 5 plus fréquents)
        if 'CLASSIFICATION_GENRE' in processed_df.columns:
            top_genres = processed_df['CLASSIFICATION_GENRE'].value_counts().nlargest(5).index.tolist()
            for genre in top_genres:
                processed_df[f'GENRE_{genre}'] = (processed_df['CLASSIFICATION_GENRE'] == genre).astype(int)

    except Exception as e:
        print(f"Erreur lors de l'encodage catégoriel: {e}")

    # === 5. CARACTÉRISTIQUES GÉOGRAPHIQUES SIMPLES ===

    try:
        # Distance depuis Paris
        processed_df['DISTANCE_FROM_PARIS'] = np.sqrt(
            (processed_df['VENUE_LATITUDE'] - DEFAULT_LAT) ** 2 +
            (processed_df['VENUE_LONGITUDE'] - DEFAULT_LON) ** 2
        ) * 111  # Conversion approximative en km

        # Clusters géographiques basiques
        if 'VENUE_STATE_CODE' in processed_df.columns:
            # Créer clusters basés sur le département
            processed_df['VENUE_STATE_CODE'] = processed_df['VENUE_STATE_CODE'].fillna('0')
            processed_df['GEO_CLUSTER'] = processed_df['VENUE_STATE_CODE'].astype(str).apply(
                lambda x: hash(x) % 10
            )
        else:
            processed_df['GEO_CLUSTER'] = 0

    except Exception as e:
        print(f"Erreur lors du traitement géographique: {e}")
        processed_df['DISTANCE_FROM_PARIS'] = 0.0
        processed_df['GEO_CLUSTER'] = 0

    # === 6. NETTOYAGE FINAL ET SÉLECTION ===

    # Identifier les colonnes à conserver
    id_cols = [col for col in ['EVENT_ID', 'VENUE_ID', 'VENUE_NAME', 'VENUE_CITY']
               if col in processed_df.columns]

    # Colonnes numériques
    num_cols = [col for col in [
        'EVENT_WEEKDAY', 'EVENT_MONTH', 'EVENT_YEAR', 'EVENT_TIME_DECIMAL',
        'DISTANCE_FROM_PARIS', 'EVENT_NAME_LENGTH', 'EVENT_INFO_LENGTH',
        'VENUE_LATITUDE', 'VENUE_LONGITUDE'
    ] if col in processed_df.columns]

    # Remplacer les NaN et valeurs infinies
    for col in num_cols:
        processed_df[col] = processed_df[col].fillna(0)
        processed_df[col] = processed_df[col].replace([np.inf, -np.inf], 0)

    # Colonnes binaires
    bin_cols = [col for col in processed_df.columns
                if col.startswith(('STATUS_', 'SEGMENT_', 'GENRE_', 'HAS_')) or col == 'GEO_CLUSTER']

    # Toutes les colonnes à conserver
    cols_to_keep = id_cols + num_cols + bin_cols

    # Sélectionner les colonnes existantes
    existing_cols = [col for col in cols_to_keep if col in processed_df.columns]
    final_df = processed_df[existing_cols]

    print(f"Dimensions finales: {final_df.shape}")
    print("Prétraitement terminé!")

    return final_df


def create_similarity_matrix_batch(df, batch_size=1000, n_neighbors=10, metric='cosine'):
    """
    Crée une matrice de similarité par lots pour économiser la mémoire

    Args:
        df (pandas.DataFrame): DataFrame prétraité
        batch_size (int): Taille des lots à traiter
        n_neighbors (int): Nombre de voisins à trouver
        metric (str): Métrique de distance ('cosine', 'euclidean', etc.)

    Returns:
        dict: Dictionnaire de recommandations
    """
    print(f"Création de la matrice de similarité (optimisée pour la mémoire)")
    print(f"Paramètres: batch_size={batch_size}, n_neighbors={n_neighbors}, metric={metric}")

    try:
        # Séparer IDs et caractéristiques
        if 'EVENT_ID' not in df.columns:
            print("Erreur: colonne EVENT_ID non trouvée")
            return {}

        id_cols = ['EVENT_ID', 'VENUE_ID', 'VENUE_NAME', 'VENUE_CITY']
        id_cols = [col for col in id_cols if col in df.columns]

        feature_cols = [col for col in df.columns if col not in id_cols]
        if not feature_cols:
            print("Erreur: pas de colonnes de caractéristiques")
            return {}

        event_ids = df['EVENT_ID'].values
        feature_matrix = df[feature_cols].values

        # Initialiser le modèle de plus proches voisins
        model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1)
        model.fit(feature_matrix)

        # Calculer les voisins par lots
        similarity_dict = {}
        num_batches = int(np.ceil(len(df) / batch_size))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_size_actual = end_idx - start_idx

            print(f"Traitement du lot {i + 1}/{num_batches} (lignes {start_idx}-{end_idx})")

            # Calculer les distances et indices pour ce lot
            batch_features = feature_matrix[start_idx:end_idx]
            distances, indices = model.kneighbors(batch_features)

            # Stocker les résultats
            for j in range(batch_size_actual):
                event_idx = start_idx + j
                event_id = event_ids[event_idx]

                # Convertir la distance en score de similarité (1 = identique, 0 = complètement différent)
                similarities = 1 / (1 + distances[j][1:])  # Ignorer le premier (lui-même)
                neighbor_indices = indices[j][1:]  # Ignorer le premier (lui-même)

                similar_events = [(event_ids[idx], float(similarities[i]))
                                  for i, idx in enumerate(neighbor_indices)]

                similarity_dict[event_id] = similar_events

        print(f"Matrice de similarité créée pour {len(similarity_dict)} événements")
        return similarity_dict

    except Exception as e:
        print(f"Erreur lors du calcul de similarité: {e}")
        return {}

def main():
    """
    Fonction principale pour exécuter le pipeline complet
    """
    try:
        # Charger les données
        print("Chargement des données...")
        df = pd.read_csv(DATASET_NAME)

        # Prétraiter les données
        processed_df = preprocess_events_data(df)

        # Sauvegarder les données prétraitées
        # processed_df.to_csv(PROCESSED_DATASET_PATH, index=False)

        # Créer la matrice de similarité par lots
        similarity_dict = create_similarity_matrix_batch(processed_df, batch_size=1000, n_neighbors=10)

        # Sauvegarder le dictionnaire de similarité au format pickle
        with open(MODEL_NAME, 'wb') as f:
            pickle.dump(similarity_dict, f)
        print("Dictionnaire de similarité sauvegardé dans 'similarity_dict.pkl'")

#########################
        # Sauvegarder également au format texte pour inspection
        #with open('../ExploratoryDataAnalysis/dataInfos/similarity_results.txt', 'w') as f:
        #    for event_id, similar_events in similarity_dict.items():
        #        f.write(f"Événement {event_id} est similaire à:\n")
        #        for similar_id, score in similar_events:
        #            f.write(f"  - {similar_id} (score: {score:.4f})\n")
        #        f.write("\n")
########################

        print("Traitement terminé avec succès!")

    except Exception as e:
        print(f"Erreur dans le processus principal: {e}")

if __name__ == "__main__":
    main()
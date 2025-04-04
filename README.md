# Système de Recommandation d'Événements

Ce projet implémente un système de recommandation d'événements culturels basé sur des techniques de similarité. Il traite un vaste jeu de données d'événements, construit une matrice de similarité et présente les recommandations via une interface utilisateur interactive.

## 1. Présentation du Dataset

### Caractéristiques du jeu de données
- **Source** : Dataset d'événements culturels en France
- **Dimensions** : 75 630 lignes × 77 colonnes
- **Contenu** : Événements culturels, spectacles, expositions, concerts, etc.
- **Portée géographique** : Principalement France, avec concentration sur Paris (62.8%)
- **Période** : Événements prévus en 2025-2026

### Distribution des types d'événements
- Arts & Théâtre : 76.7%
- Divers : 13.6%
- Musique : 9.0%
- Films : 0.4%
- Sports : 0.2%

### Principales colonnes utilisées
- `EVENT_ID` : Identifiant unique de l'événement
- `EVENT_NAME` : Nom de l'événement
- `EVENT_INFO` : Description détaillée
- `CLASSIFICATION_SEGMENT/GENRE/SUB_GENRE` : Catégorisation
- `VENUE_NAME` et données géographiques : Lieu et localisation
- `EVENT_START_LOCAL_DATE/TIME` : Informations temporelles
- `EVENT_IMAGE_URL` : Lien vers l'image de l'événement

## 2. Problèmes Identifiés et Solutions

### Problèmes dans le dataset

#### 1. Valeurs manquantes
- **Problème** : Nombreuses colonnes avec valeurs manquantes (ex: 80.3% pour les attributs liés aux attractions)
- **Solution** : 
  - Création d'indicateurs binaires pour signaler la présence/absence (ex: `HAS_ATTRACTION`)
  - Remplissage avec valeurs par défaut appropriées
  - Suppression des colonnes entièrement vides ou presque vides

#### 2. Colonnes redondantes
- **Problème** : Plusieurs paires de colonnes avec informations similaires ou dupliquées
- **Solution** : Suppression des colonnes redondantes, conservation des versions les plus informatives

#### 3. Déséquilibre des données
- **Problème** : Surreprésentation des événements de type "Arts & Théâtre" (76.7%)
- **Solution** : Prise en compte équilibrée des différentes catégories dans le calcul de similarité

#### 4. Données textuelles non structurées
- **Problème** : Descriptions longues et non standardisées
- **Solution** : Extraction de caractéristiques simples (longueur) et encodage des catégories principales

### Défis techniques

#### 1. Problèmes de mémoire
- **Problème** : Calcul de similarité entre 75 630 événements dépassant la mémoire disponible
- **Solution** : 
  - Traitement par lots (batch processing)
  - Utilisation de NearestNeighbors de scikit-learn
  - Optimisation des paramètres de calcul

#### 2. Erreurs de calcul
- **Problème** : Erreurs "Mean of empty slice" lors du traitement des données
- **Solution** : 
  - Gestion robuste des valeurs manquantes
  - Filtrage des valeurs infinies ou invalides
  - Vérifications systématiques avant les calculs

#### 3. Incompatibilités de type
- **Problème** : Colonnes avec types mixtes
- **Solution** : Chargement avec `low_memory=False` et conversion explicite des types

## 3. Étapes du Projet

### Étape 1 : Analyse exploratoire et prétraitement
1. Analyse des dimensions et distribution du dataset
2. Identification des valeurs manquantes et aberrantes
3. Suppression des colonnes inutiles ou redondantes
4. Création de caractéristiques dérivées :
   - Caractéristiques temporelles (jour de semaine, mois, année)
   - Caractéristiques géographiques (distance depuis Paris)
   - Indicateurs binaires pour catégories

### Étape 2 : Construction du système de recommandation
1. Préparation des caractéristiques pour le calcul de similarité
2. Implémentation du calcul optimisé par lots avec NearestNeighbors
3. Calcul des scores de similarité (cosinus, euclidienne ou Jaccard)
4. Sauvegarde des résultats dans une structure de données efficace

### Étape 3 : Développement de l'interface utilisateur
1. Conception d'une application Streamlit intuitive
2. Intégration des données originales et de la matrice de similarité
3. Implémentation des fonctionnalités de recherche et filtrage
4. Création de l'affichage des événements et recommandations

## 4. Fonctionnalités de l'Application Streamlit

### Navigation et recherche
- **Filtrage par catégorie** : Sélection parmi les segments principaux (Arts & Théâtre, Musique, etc.)
- **Filtrage par ville** : Recherche géographique des événements
- **Recherche textuelle** : Recherche par mot-clé dans les noms d'événements
- **Affichage des résultats** : Compteur dynamique du nombre d'événements trouvés

### Visualisation des événements
- **Affichage détaillé** : Présentation complète de l'événement sélectionné
- **Informations principales** : Nom, lieu, date, catégorie
- **Média** : Affichage de l'image de l'événement quand disponible
- **Description** : Texte complet accessible via un expander
- **Liens** : Accès direct à la page officielle de l'événement

### Système de recommandation
- **Top 10 des événements similaires** : Présentation des événements les plus proches
- **Affichage en grille** : Visualisation claire des recommandations
- **Scores de similarité** : Classement par pertinence
- **Exploration facile** : Navigation intuitive entre les recommandations

### Interface adaptative
- **Design responsive** : Adaptation à différentes tailles d'écran
- **Feedback utilisateur** : Informations claires sur les résultats de recherche
- **Gestion des erreurs** : Robustesse face aux données manquantes ou problématiques

## 5. Instructions d'Utilisation
### Installation de dépendances
```bash
pip install -r requirements.txt
```    
### Prétraitement et création de la matrice de similarité
```bash
python similarity_processing.py
```
Génère le fichier `similarity_dict.pkl`

### Lancement de l'application Streamlit
```bash
streamlit run streamlit_app.py
```

### Utilisation de l'application
1. Utilisez les filtres dans la barre latérale pour affiner votre recherche
2. Sélectionnez un événement dans la liste déroulante
3. Explorez les détails de l'événement sélectionné
4. Découvrez les événements similaires recommandés en dessous

## 6. Perspectives d'Amélioration

- **Personnalisation des recommandations** : Intégration des préférences utilisateur
- **Enrichissement avec données externes** : Météo, transports, popularité sur réseaux sociaux
- **Recommandation contextuelle** : Prise en compte de la localisation et disponibilité de l'utilisateur
- **Interface mobile** : Version optimisée pour appareils mobiles
- **Analyse temporelle** : Recommandations basées sur périodes similaires ou complémentaires
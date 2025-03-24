import pandas as pd

# Charger le dataset (modifie le chemin selon ton fichier)
file_path = "../datasets/fr-events.csv"
df = pd.read_csv(file_path)  # Remplace par pd.read_json(file_path) si JSON

# Définir le fichier de sortie
output_file = "dataInfos/rapport_dataset.txt"

# Ouvrir le fichier en mode écriture
with open(output_file, "w", encoding="utf-8") as f:
    # Aperçu du dataset
    f.write("\n🔍 Aperçu du dataset :\n")
    f.write(df.head().to_string() + "\n")

    # Dimensions du dataset
    f.write("\n📏 Dimensions du dataset :\n")
    f.write(f"Nombre de lignes : {df.shape[0]}\n")
    f.write(f"Nombre de colonnes : {df.shape[1]}\n")

    # Types des colonnes
    f.write("\n🛠️ Types des colonnes :\n")
    f.write(df.dtypes.to_string() + "\n")

    # Valeurs manquantes
    f.write("\n⚠️ Valeurs manquantes par colonne :\n")
    f.write(df.isnull().sum().to_string() + "\n")

    # Doublons
    f.write("\n🔄 Nombre de doublons :\n")
    f.write(str(df.duplicated().sum()) + "\n")

    # Statistiques descriptives des variables numériques
    f.write("\n📊 Statistiques descriptives (variables numériques) :\n")
    f.write(df.describe().to_string() + "\n")

    # Statistiques descriptives des variables catégorielles
    f.write("\n🔠 Statistiques sur les variables catégoriques :\n")
    f.write(df.describe(include=['object']).to_string() + "\n")

    # Nombre de valeurs uniques par colonne
    f.write("\n🔢 Nombre de valeurs uniques par colonne :\n")
    f.write(df.nunique().to_string() + "\n")

    # Valeurs les plus fréquentes pour chaque colonne catégorielle
    f.write("\n🏆 Valeurs les plus fréquentes pour chaque variable catégorielle :\n")
    for col in df.select_dtypes(include=['object']).columns:
        f.write(f"\nTop valeurs pour {col} :\n")
        f.write(df[col].value_counts().head(5).to_string() + "\n")

    # Résumé global du dataset
    f.write("\n📋 Résumé global du dataset :\n")
    df.info(buf=f)

print(f"✅ Rapport généré avec succès : {output_file}")

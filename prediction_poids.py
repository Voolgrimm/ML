# Importation des bibliothèques nécessaires
import pandas as pd  # Pour manipuler les données sous forme de tableaux (DataFrames)
import numpy as np   # Pour les opérations mathématiques et manipulation de tableaux
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression  # Modèle de régression linéaire

# Chargement des données à partir d'un fichier CSV
# On suppose que le fichier contient des colonnes : 'age', 'poids', 'taille', 'sexe'
df = pd.read_csv("data/age_vs_poids_vs_taille_vs_sexe.csv")

# **Définition des variables pour le modèle :**
# X = variables explicatives (ou prédictives) : 'age', 'taille', 'sexe'
X = df[['age', 'taille', 'sexe']]  # Ces colonnes serviront à prédire le poids
# y = variable cible : 'poids'
y = df.poids  # C'est la valeur que l'on veut prédire (le poids)

# **Création et entraînement du modèle :**
# Initialisation du modèle de régression linéaire
reg = LinearRegression()

# Entraînement du modèle sur les données
# fit() ajuste le modèle en trouvant les relations entre X et y
reg.fit(X, y)

# **Évaluation du modèle :**
# Affichage du score R², qui mesure la qualité de la prédiction (1.0 = parfait)
print(f"Score R² du modèle : {reg.score(X, y):.2f}")

# Affichage des coefficients du modèle
# Les coefficients indiquent l'importance de chaque variable explicative
print(f"Coefficients du modèle : {reg.coef_}")

# **Prédiction d'une nouvelle donnée :**
# Création d'une nouvelle observation (par exemple : age=153, taille=150 cm, sexe=0)
# On suppose ici que 'sexe' est codé en 0 pour "homme" et 1 pour "femme"
data = pd.DataFrame([[153, 150, 0]], columns=["age", "taille", "sexe"])

# Prédiction du poids pour cette observation
poids = reg.predict(data)
print(f"Le poids prévu pour cette personne est de {poids[0]:.2f} kg")

# **Calcul des erreurs du modèle :**
# Prédictions du modèle sur l'ensemble des données initiales
y_pred = reg.predict(X)

# Calcul et affichage de plusieurs métriques d'évaluation
# 1. Erreur quadratique moyenne (MSE) : moyenne des carrés des écarts entre y et y_pred
mse = mean_squared_error(y, y_pred)
print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")

# 2. Erreur absolue moyenne (MAE) : moyenne des écarts absolus entre y et y_pred
mae = mean_absolute_error(y, y_pred)
print(f"Erreur absolue moyenne (MAE) : {mae:.2f}")

# 3. Erreur en pourcentage absolu moyenne (MAPE) : mesure relative des erreurs en pourcentage
mape = mean_absolute_percentage_error(y, y_pred)
print(f"Erreur en pourcentage absolu moyenne (MAPE) : {mape:.2%}")

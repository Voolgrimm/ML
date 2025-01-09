# Importation des bibliothèques nécessaires
import pandas as pd  # Pandas est utilisé pour manipuler les données sous forme de tables
import numpy as np  # Numpy pour les opérations mathématiques sur les tableaux
import seaborn as sns  # Seaborn pour la visualisation graphique des données
import matplotlib.pyplot as plt  # Matplotlib pour dessiner des graphiques
from sklearn.linear_model import LinearRegression  # Pour appliquer la régression linéaire
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error  # Pour calculer les erreurs de prédiction
from sklearn.model_selection import train_test_split  # Pour séparer les données en jeu d'entraînement et test
from sklearn.preprocessing import MinMaxScaler  # Pour normaliser les données (les ramener entre 0 et 1)

# Charger les données à partir du fichier CSV
df = pd.read_csv("data/advertising.csv")  # Chargement du fichier de données (fichier CSV)

# Afficher les premières lignes des données pour une première inspection
print(df.head())  # Affiche les 5 premières lignes du DataFrame pour avoir une idée de la structure

# Afficher des statistiques descriptives sur les données (moyenne, écart-type, min, max, etc.)
print(df.describe())  # Affiche des statistiques comme la moyenne, écart-type des colonnes numériques

# Afficher des informations générales sur le DataFrame (types de colonnes, valeurs manquantes, etc.)
print(df.info())  # Donne un aperçu des types de données et si certaines colonnes contiennent des valeurs manquantes

# Création de graphiques pour visualiser la relation entre chaque variable et les ventes
fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Créer un ensemble de 3 graphiques sur une même ligne

# Premier graphique : Relation entre TV et les ventes
plt.subplot(1, 3, 1)  # Premier graphique sur la première position (1 ligne, 3 colonnes)
sns.regplot(x=df[['tv']], y=df.ventes)  # Crée un graphique de régression entre TV et les ventes
plt.ylabel('ventes')  # Nom de l'axe des ordonnées (ventes)
plt.xlabel('TV')  # Nom de l'axe des abscisses (publicité à la télévision)
plt.title('ventes = a * tv + b')  # Titre du graphique
plt.grid()  # Affiche une grille sur le graphique
sns.despine()  # Retire les bordures du graphique

# Deuxième graphique : Relation entre Radio et les ventes
plt.subplot(1, 3, 2)  # Deuxième graphique
sns.regplot(x=df[['radio']], y=df.ventes)  # Graphique de régression entre Radio et les ventes
plt.ylabel('ventes')
plt.xlabel('radio')
plt.title('ventes = a * radio + b')
plt.grid()
sns.despine()

# Troisième graphique : Relation entre Journaux et les ventes
plt.subplot(1, 3, 3)  # Troisième graphique
sns.regplot(x=df[['journaux']], y=df.ventes)  # Graphique de régression entre Journaux et les ventes
plt.ylabel('ventes')
plt.xlabel('journaux')
plt.title('ventes = a * journaux + b')
plt.grid()
sns.despine()

# Afficher les graphiques
plt.tight_layout()  # Organise l'espace entre les graphiques pour éviter qu'ils ne se chevauchent
plt.show()  # Affiche les graphiques

# Afficher la matrice de corrélation pour analyser les relations entre les variables
print(df.corr())  # Affiche une matrice de corrélation entre toutes les variables du DataFrame

# Définir le modèle de régression linéaire
reg = LinearRegression()  # Crée un modèle de régression linéaire

# Définir les variables indépendantes (X) et la variable cible (y)
X = df[["tv", "radio", "journaux"]]  # Variables explicatives (les facteurs influençant les ventes)
y = df["ventes"]  # Variable cible (les ventes à prédire)

# Diviser les données en ensemble d'entraînement (80%) et ensemble de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% pour l'entraînement, 20% pour les tests

# Entraîner le modèle de régression sur l'ensemble d'entraînement
reg.fit(X_train, y_train)  # Entraîne le modèle avec les données d'entraînement

# Prédire les valeurs de "ventes" pour l'ensemble de test
y_pred_test = reg.predict(X_test)  # Utilise le modèle pour prédire les ventes sur les données de test

# Calculer et afficher les erreurs de prédiction (RMSE et MAPE)
print(f"RMSE: {mean_squared_error(y_test, y_pred_test)}")  # Erreur quadratique moyenne (RMSE) entre prédictions et vraies valeurs
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")  # Erreur absolue moyenne en pourcentage (MAPE)

# Création de la variable quadratique tv2 pour capturer des effets non linéaires de la variable tv
df['tv2'] = df['tv']**2  # Ajoute une nouvelle colonne tv2 qui est le carré de la variable TV

# Définir les variables à utiliser pour la régression, y compris la nouvelle variable tv2
variables = ['tv', 'radio', 'journaux', 'tv2']  # Les variables à inclure dans le modèle

# Normalisation des données pour mettre toutes les variables sur une échelle de 0 à 1
scaler = MinMaxScaler()  # Crée un objet pour normaliser les données

# Appliquer la normalisation sur les variables sélectionnées
data_array = scaler.fit_transform(df[variables])  # Applique la normalisation aux données sélectionnées

# Transformer le tableau numpy en DataFrame pour garder les noms de colonnes
df_scaled = pd.DataFrame(data_array, columns=variables)  # Crée un DataFrame avec les données normalisées

# Vérifier que les données ont été correctement normalisées (les min et max doivent être 0 et 1)
print(df_scaled.describe().loc[['min', 'max']])  # Affiche les valeurs minimales et maximales pour vérifier la normalisation

# Redéfinir les variables indépendantes (X) et la variable cible (y) avec les données normalisées
X = df_scaled[variables]  # Variables explicatives avec les données normalisées
y = df.ventes  # Variable cible (ventes)

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un nouveau modèle de régression linéaire
reg = LinearRegression()  # Crée un modèle de régression linéaire

# Entraîner le modèle sur les données normalisées
reg.fit(X_train, y_train)  # Entraîne le modèle avec les données normalisées

# Prédire les ventes sur l'ensemble de test
y_pred_test_scaled = reg.predict(X_test)  # Prédictions sur les données de test

# Afficher les résultats de la régression sur les données normalisées
print(f"-- Régression ventes ~ radio + journaux + tv + tv^2")
print(f"\tRMSE: {mean_squared_error(y_test, y_pred_test_scaled)}")  # Affiche l'erreur quadratique moyenne
print(f"\tMAPE: {mean_absolute_percentage_error(y_test, y_pred_test_scaled)}")  # Affiche l'erreur absolue moyenne

# ** Comparaison des 3 regressions **

# Créer une nouvelle variable 'tv_radio' qui est l'interaction entre TV et Radio
df['tv_radio'] = df.tv * df.radio  # Multiplie les colonnes TV et Radio pour créer une nouvelle variable croisée

# Dictionnaire pour stocker les différents types de régressions à tester
regressions = {
    'simple : y ~ tv + radio + journaux' : ['tv','radio','journaux'],  # Modèle simple avec TV, Radio et Journaux
    'quadratique : y ~ tv + radio + journaux + tv2' : ['tv','radio','journaux','tv2'],  # Modèle avec TV au carré
    'croisée : y ~ tv + radio + journaux + tv*radio' : ['tv','radio','journaux','tv_radio']  # Modèle avec l'interaction TV*Radio
}

# Variable cible toujours la même (ventes)
y = df.ventes

# Boucle pour tester chaque modèle de régression
for title, variables in regressions.items():
    # Normalisation des variables explicatives
    scaler = MinMaxScaler()
    data_array = scaler.fit_transform(df[variables])  # Normalisation des variables

    # Séparation des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data_array, y, test_size=0.2, random_state=42)

    # Créer un modèle de régression linéaire
    reg = LinearRegression()

    # Entraîner le modèle
    reg.fit(X_train, y_train)

    # Prédire les ventes
    y_pred_test = reg.predict(X_test)

    # Afficher les résultats de chaque régression
    print(f"\n-- {title}")
    print(f"\tRMSE: {mean_squared_error(y_test, y_pred_test)}")  # Affiche l'erreur quadratique moyenne pour chaque modèle
    print(f"\tMAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")  # Affiche l'erreur absolue moyenne

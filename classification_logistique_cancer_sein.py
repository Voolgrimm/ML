# ** Importation des bibliothèques nécessaires **
# - `load_breast_cancer` : pour charger le jeu de données sur le cancer du sein.
# - `pyplot` et `sns` : pour visualiser les données.
# - `train_test_split` : pour séparer les données en ensembles d'entraînement et de test.
# - `LogisticRegression` : pour créer et entraîner le modèle de régression logistique.
# - `accuracy_score` et `confusion_matrix` : pour évaluer les performances du modèle.
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix



# ** Chargement des données **
# `X` contient les données (caractéristiques) et `y` contient les étiquettes (diagnostics : 0 ou 1).
X, y = load_breast_cancer(return_X_y=True)
print("Dimensions des données :", X.shape)  # Affiche le nombre de lignes et de colonnes.

# ** Visualisation des classes **
# Montre les classes disponibles (0 pour bénin, 1 pour malin).
print("Classes disponibles :", set(y))

# ** Séparation des données en ensembles d'entraînement et de test **
# - `test_size=0.2` : 20% des données pour le test, 80% pour l'entraînement.
# - `random_state=42` : garantit que la séparation est reproductible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ** Création et entraînement du modèle de régression logistique **
# - `random_state=808` : pour des résultats reproductibles.
# - `max_iter=10000` : augmente le nombre d'itérations pour éviter un échec de convergence.
clf = LogisticRegression(random_state=808, max_iter=10000).fit(X_train, y_train)

# ** Prédictions sur des échantillons spécifiques **
# - `clf.predict()` prédit la classe (0 ou 1).
# - `clf.predict_proba()` donne la probabilité pour chaque classe.
print("Prédiction pour l'échantillon 8 :", clf.predict([X[8, :]])[0])
print("Probabilité pour l'échantillon 8 :", clf.predict_proba([X[8, :]])[0][0])
print("Prédiction pour l'échantillon 13 :", clf.predict([X[13, :]])[0])
print("Probabilité pour l'échantillon 13 :", clf.predict_proba([X[13, :]])[0][1])

# ** Histogramme des probabilités prédites pour la classe positive **
# Montre la distribution des probabilités pour les prédictions sur l'ensemble de test.
y_hat_proba = clf.predict_proba(X_test)[:, 1]
sns.histplot(y_hat_proba, kde=True, bins=30, color='blue')
plt.title("Distribution des probabilités pour la classe positive")
plt.xlabel("Probabilité")
plt.ylabel("Fréquence")
plt.show()

# ** Mesure de l'exactitude (accuracy) **
# Pourcentage de prédictions correctes.
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude (accuracy) :", accuracy)

# ** Matrice de confusion **
# Affiche les vrais positifs, vrais négatifs, faux positifs et faux négatifs.
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)

# Visualisation de la matrice de confusion.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.title("Matrice de Confusion")
plt.show()

# ** Évaluation avec différents seuils **
# On teste l'impact de seuils différents (0.3 et 0.7) pour classifier les probabilités.
y_pred_03 = [0 if value < 0.3 else 1 for value in y_hat_proba]
y_pred_07 = [0 if value < 0.7 else 1 for value in y_hat_proba]

# Matrice de confusion pour le seuil 0.3.
cm_03 = confusion_matrix(y_test, y_pred_03)
print("Matrice de confusion (seuil 0.3) :\n", cm_03)

# Matrice de confusion pour le seuil 0.7.
cm_07 = confusion_matrix(y_test, y_pred_07)
print("Matrice de confusion (seuil 0.7) :\n", cm_07)

# ** Autres métriques de classification **
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Précision : proportion de prédictions positives correctes.
print("Précision :", precision_score(y_test, y_pred))

# Rappel : proportion de cas positifs correctement détectés.
print("Rappel :", recall_score(y_test, y_pred))

# ROC-AUC : performance globale du modèle pour distinguer les classes.
print("ROC-AUC :", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# ** Courbe ROC **
from sklearn.metrics import roc_curve

# Courbe ROC : relation entre le taux de vrais positifs et de faux positifs.
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, color='blue', label='Courbe ROC')
plt.grid(True)
plt.title('Courbe ROC')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.legend()
plt.show()


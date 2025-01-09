# Importation des bibliothèques nécessaires pour charger, entraîner, évaluer, et visualiser les données
from sklearn.datasets import load_iris  # Permet de charger le jeu de données Iris
from sklearn.model_selection import train_test_split  # Utilisé pour séparer les données en entraînement et test
from sklearn.linear_model import LogisticRegression  # Modèle de régression logistique
import seaborn as sns  # Bibliothèque pour la visualisation
from matplotlib import pyplot as plt  # Bibliothèque pour la visualisation des graphiques
from sklearn.metrics import accuracy_score, confusion_matrix  # Mesures de la performance du modèle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve  # Autres mesures de performance
from sklearn.model_selection import cross_val_score  # Permet la validation croisée pour évaluer le modèle
from sklearn.metrics import classification_report

# Chargement du dataset Iris : il contient des informations sur les fleurs et leurs caractéristiques
X, y = load_iris(return_X_y=True)

# Affichage de la dimension des données (nombre d'échantillons et de caractéristiques)
print("dimension des données : \n", X.shape)

# Affichage des classes disponibles dans le dataset (types de fleurs)
print("classes des données : \n", set(y))

# Séparation des données en 80% pour l'entraînement et 20% pour le test (utilisation du random_state pour reproductibilité)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Création et entraînement du modèle de régression logistique (max_iter=10000 pour donner plus de temps d'apprentissage)
clf = LogisticRegression(random_state=150, max_iter=10000).fit(X_train, y_train)

# Prédiction pour deux échantillons spécifiques, en affichant les classes et les probabilités de chaque classe
print("Prédiction pour l'échantillon 10 :", clf.predict([X[10, :]])[0])
print("Proba pour l'échantillon 10 :", clf.predict_proba([X[10, :]])[0][0])
print("Prédiction pour l'échantillon 50 :", clf.predict([X[50, :]])[0])
print("Proba pour l'échantillon 50 :", clf.predict_proba([X[50, :]])[0][1])

# Calcul des probabilités prédites sur le jeu de test et visualisation des probabilités pour la classe 1 (toxic)
y_hat_proba = clf.predict_proba(X_test)[:,1]
sns.histplot(y_hat_proba, kde=True, bins = 30, color = 'blue')
plt.title("Distribution des proba prédites pour la Classe d'iris toxique")
plt.xlabel("Probabilité prédite")
plt.ylabel("nombre d'échantillons")
plt.show()

# Mesure de la précision globale du modèle en calculant l'accuracy sur le jeu de test
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)

# Calcul de la matrice de confusion, qui montre le nombre de bonnes et mauvaises prédictions pour chaque classe
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)

# Visualisation de la matrice de confusion avec une heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vérité')
plt.show()

# Évaluation du modèle avec différents seuils de probabilité (0.1, 0.3, 0.7, 0.9)
y_pred_01 = [0 if value < 0.1 else 1.0 for value in y_hat_proba]
y_pred_03 = [0 if value < 0.3 else 1.0 for value in y_hat_proba]
y_pred_07 = [0 if value < 0.7 else 1.0 for value in y_hat_proba]
y_pred_09 = [0 if value < 0.9 else 1.0 for value in y_hat_proba]

# Calcul des matrices de confusion pour chaque seuil
cm_01 = confusion_matrix(y_test, y_pred_01)
cm_03 = confusion_matrix(y_test, y_pred_03)
cm_07 = confusion_matrix(y_test, y_pred_07)
cm_09 = confusion_matrix(y_test, y_pred_09)

# Affichage des matrices de confusion pour chaque seuil
print("Matrice de confusion pour seuil 0.1 :\n", cm_01)
print("Matrice de confusion pour seuil 0.3 :\n", cm_03)
print("Matrice de confusion pour seuil 0.7 :\n", cm_07)
print("Matrice de confusion pour seuil 0.9 : \n", cm_09)

# Calcul et affichage de la précision pour chaque classe
print("Précision par classe : ", precision_score(y_test, y_pred, average=None))

# Calcul du rappel (sensitivity) pour un problème multiclasses
print("Rappel :", recall_score(y_test, y_pred, average='macro'))

# Calcul du score ROC-AUC, qui est une mesure de la performance du modèle pour plusieurs classes
print("ROC-AUC (micro) :", roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr', average='micro'))

# Tracé des courbes ROC pour chaque classe
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 0], pos_label=0)
plt.plot(fpr, tpr, color='blue', label='Classe 0')
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)
plt.plot(fpr, tpr, color='green', label='Classe 1')
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 2], pos_label=2)
plt.plot(fpr, tpr, color='red', label='Classe 2')

# Affichage de la courbe ROC
plt.grid(True)
plt.title('Courbe ROC pour chaque classe')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.legend()
plt.show()

# Validation croisée pour évaluer la performance du modèle sur plusieurs partitions des données
cv_scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation
print("Scores de validation croisée : ", cv_scores)
print("Moyenne des scores : ", cv_scores.mean())

# Affichage du rapport de classification avec des étiquettes et des métriques
print(classification_report(y_pred, y_test))


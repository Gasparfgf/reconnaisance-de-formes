# Reconnaissance de Formes — Classification de Feuilles d'Arbres

## Description

Ce projet s'inscrit dans le cadre d'un tp universitaire dont l'objectif est de développer une **application de reconnaissance de formes** basée sur des **caractéristiques géométriques** extraites d’images.
L’application identifie automatiquement la **catégorie d’un type de feuille** (ex. chêne, érable, bouleau, saule) à partir d’une photo.

Le projet a été réalisé **sans réseau de neurones**, en utilisant uniquement des **méthodes classiques de vision par ordinateur** (OpenCV) et de **machine learning** (Random Forest).

---

## Organisation du projet

```code
│
├── dataset/ # Contains the used images
│ ├── chene/
│ ├── erable/
│ ├── bouleau/
│ └── saule/
│
├── test_images/
│ ├── test_model.py
│ └── 1568.jpg
│
├── feuilles_features.csv # Dataset généré automatiquement
├── modele_feuilles_rf.pkl # Modèle entraîné sauvegardé
├── scaler.pkl # Scaler utilisé pour normaliser les données
├── main.py # Script principal
└── README.md # Documentation technique
```

## Technologies utilisées

| Technologie | Rôle |
|--------------|------|
| **Python 3.x** | Langage principal |
| **OpenCV** | Traitement et analyse d’images |
| **NumPy / Pandas** | Manipulation et structuration des données |
| **Scikit-learn** | Entraînement du modèle et évaluation |
| **Matplotlib / Seaborn** | Visualisations et matrices de confusion |
| **Joblib** | Sauvegarde du modèle et du scaler |

---

## Fonctionnalités principales

- Chargement et traitement automatique des images
- Extraction des contours et caractéristiques de forme
- Génération d’un jeu de données (`feuilles_features.csv`)
- Entraînement d’un modèle de classification (Random Forest)
- Évaluation des performances (précision, rappel, F1-score)
- Prédiction sur de nouvelles images

---

## Installation

```bash
# Cloner le projet
git clone git@github.com:Gasparfgf/reconnaisance-de-formes.git
cd reconnaisance-de-formes

# installer les dépendences
pip install opencv-python numpy pandas scikit-learn matplotlib seaborn joblib
```

## Utilisation

```bash
python main.py
```

### Exemples de sortie :

1. Générer le dataset

Assurez-vous que les images sont bien classées dans dataset/<nom_classe>/ :

```luma
Nombre total d'images traitées : 60
Dataset sauvegardé dans feuilles_features.csv
```

### 2. Entraîner et sauvegarder le modèle

Lors de la première exécution, le modèle est automatiquement entraîné :

```luma
=== Entraînement du modèle ===

Précision du modèle : 83.33%
...
Modèle et scaler sauvegardés !
```

3. Tester le modèle avec une nouvelle image

Place une image dans le dossier test_images/ (ex. nouvelle_feuille.jpg), puis relance le script.

Exemple de sortie :

```luma
Cette feuille est probablement : SAULE

```

---

## Auteur

**Nom** : [Gaspar da Rosa Francisco](https://github.com/Gasparfgf)
**Année universitaire** : M2 - ALT - 2025
**Projet** : Reconnaissance de formes
**Langage** : Python
**Encadrant** : Rémy Kessler

---

## Crédits

Les images proviennent du jeu de données “Leaf Classification” disponible sur [Kaggle](https://leafsnap.com/dataset/).
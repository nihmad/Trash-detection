# Système d'aide au tri intelligent des déchets

Classification automatique de déchets recyclables par vision par ordinateur.
Le projet compare deux approches : un **CNN personnalisé** entraîné de zéro
et un modèle de **transfert d'apprentissage basé sur ResNet50**.

Le projet peut s'exécuter de deux façons : **en local** (installation classique)
ou **dans le cloud via Google Colab**, sans rien installer sur sa machine.

## Classes prédites

Cinq classes harmonisées à partir du dataset TrashNet :

| Classe (dossier) | Libellé |
|------------------|---------|
| `cardboard`      | Carton  |
| `glass`          | Verre   |
| `metal`          | Métal   |
| `paper`          | Papier  |
| `plastic`        | Plastique |

La classe `trash` de TrashNet est volontairement exclue : elle est
hétérogène (déchets divers non recyclables) et n'entre pas dans le périmètre
des cinq matières recyclables visées. L'harmonisation est documentée dans
`src/dataset.py` (`LABEL_MAPPING`).

## Dataset

Le dataset TrashNet est disponible sur Kaggle :
https://www.kaggle.com/datasets/feyzazkefe/trashnet

Deux façons de le récupérer.

**Téléchargement manuel (usage local).** Téléchargez l'archive depuis Kaggle,
décompressez-la, puis organisez les dossiers de classes ainsi :

```
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
└── plastic/
```

**Téléchargement automatique avec kagglehub (recommandé en Colab).**

```python
import kagglehub
path = kagglehub.dataset_download("feyzazkefe/trashnet")
print("Données téléchargées dans :", path)
```

Dans ce cas, les images restent dans le cache kagglehub : il suffit de
passer ce chemin via l'option `--raw-dir` des scripts (voir plus bas). Les
images ne sont jamais versionnées (voir `.gitignore`).

## Structure du projet

```
project/
├── data/
│   ├── raw/          # Images sources (un dossier par classe)
│   └── processed/    # Données intermédiaires éventuelles
├── notebooks/
│   ├── 01_exploration.ipynb   # Exploration du dataset
│   └── colab_pipeline.ipynb   # Pipeline complet prêt pour Google Colab
├── src/
│   ├── preprocessing.py  # Pipeline OpenCV
│   ├── dataset.py        # Indexation, harmonisation, split, tf.data
│   ├── cnn_model.py      # CNN personnalisé
│   ├── resnet_model.py   # ResNet50 (transfert + fine-tuning)
│   ├── train.py          # Entraînement (CLI)
│   ├── evaluate.py       # Métriques + analyse d'erreurs
│   └── utils.py          # Chemins, graines, logging, IO
├── app/
│   └── streamlit_app.py  # Application de démonstration
├── models/           # Modèles entraînés (.keras)
├── reports/
│   ├── figures/      # Courbes, matrices de confusion, erreurs
│   └── metrics/      # Historiques et rapports JSON
├── tests/            # Tests unitaires (pytest)
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Exécution sur Google Colab (sans installation locale)

Le notebook `notebooks/colab_pipeline.ipynb` enchaîne toutes les étapes du
projet dans le cloud, avec un GPU gratuit et aucune installation locale. Il
ne contient pas la logique du projet : il **orchestre** les modules `src/`,
qui restent la source de vérité.

Étapes :

1. Ouvrir le notebook dans Colab (`Fichier → Importer un notebook`).
2. Activer le GPU : `Exécution → Modifier le type d'exécution → GPU (T4)`.
3. Renseigner l'URL de son dépôt GitHub dans la première cellule.
4. Exécuter les cellules de haut en bas.

Le notebook clone le dépôt, télécharge le dataset avec kagglehub, localise
automatiquement le dossier des classes, entraîne et évalue les deux modèles,
affiche les figures, puis permet de tester le modèle sur ses propres images.
Une cellule optionnelle sauvegarde les modèles et rapports dans Google Drive
(la session Colab étant éphémère).

## Installation locale

Python 3.11 ou supérieur est requis.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Prétraitement

Le pipeline OpenCV (`src/preprocessing.py`) enchaîne :

1. **Vérification des images corrompues** — toute image illisible ou au
   format non supporté est écartée à l'indexation.
2. **Conversion de format** — lecture BGR (OpenCV) puis conversion en RGB.
3. **Redimensionnement** — taille cible 128×128 (interpolation par aire).
4. **Normalisation** — pixels ramenés dans l'intervalle [0, 1].
5. **Augmentation de données** — rotation, zoom, translation et flip
   horizontal, appliqués uniquement à l'entraînement.

La normalisation propre à chaque modèle est intégrée au modèle lui-même :
`Rescaling(1/255)` pour le CNN, `preprocess_input` ImageNet pour ResNet50.

## Entraînement

```bash
# CNN personnalisé
python -m src.train --model cnn --epochs 30

# ResNet50 : entraînement de la tête puis fine-tuning
python -m src.train --model resnet --epochs 15 --fine-tune-epochs 10
```

Si les données viennent de kagglehub, ajouter l'option `--raw-dir` avec le
chemin renvoyé (ou le sous-dossier contenant les classes) :

```bash
python -m src.train --model cnn --epochs 30 --raw-dir "/chemin/vers/les/classes"
```

Chaque entraînement produit :
- le modèle sauvegardé dans `models/<modèle>.keras` ;
- l'historique dans `reports/metrics/<modèle>_history.json` ;
- les courbes dans `reports/figures/<modèle>_training_curves.png`.

## Évaluation et analyse des erreurs

```bash
python -m src.evaluate --model cnn
python -m src.evaluate --model resnet
```

L'option `--raw-dir` s'applique de la même façon qu'à l'entraînement.

Génère pour chaque modèle :
- les métriques globales (accuracy, precision, recall, F1) ;
- le rapport de classification détaillé par classe ;
- la matrice de confusion (`reports/figures/`) ;
- une planche d'exemples mal classés avec score de confiance ;
- les paires de classes les plus confondues.

Tous les résultats chiffrés sont enregistrés dans
`reports/metrics/<modèle>_evaluation.json`.

## Application de démonstration

```bash
streamlit run app/streamlit_app.py
```

L'application permet de charger une image, de choisir le modèle, puis
d'afficher la classe prédite, le score de confiance et la distribution des
probabilités. En environnement notebook (Colab), la dernière cellule du
pipeline offre la même fonction de test directement dans le notebook.

## Tests

```bash
pytest
```

Les tests couvrent le prétraitement et la gestion du dataset à partir
d'images synthétiques ; ils ne nécessitent ni le dataset complet ni un GPU.

## Comparaison des deux approches

Le CNN personnalisé est léger et rapide à entraîner sur une machine
personnelle, mais dispose d'une capacité de généralisation limitée sur un
jeu de taille modeste. ResNet50 en transfert d'apprentissage réutilise des
représentations apprises sur ImageNet et atteint généralement de meilleures
performances, au prix d'un modèle plus lourd. La comparaison chiffrée se lit
dans les fichiers `reports/metrics/*_evaluation.json` après évaluation des
deux modèles.

## Licence

Distribué sous licence MIT. Voir le fichier `LICENSE`.

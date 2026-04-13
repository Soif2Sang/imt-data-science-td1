# TD1 Plantar Activity Recognition

Projet pedagogique pour analyser des donnees de semelles instrumentees et entrainer un modele de reconnaissance d'activites.

Le repo contient :

- un script d'entrainement robuste avec protocole `Train / Validation / Test`
- un notebook de cours interactif pour comprendre le pipeline pas a pas
- un notebook d'exploration du TD
- des scripts utilitaires d'agregation et de verification des donnees

## Structure

```text
.
├── data/
│   └── README.md
├── docs/
│   ├── TD.md
│   └── observations.md
├── notebooks/
│   ├── cours_train_plantar_model.ipynb
│   └── td_notebook.ipynb
├── scripts/
│   ├── aggregate_td_data.py
│   ├── td_data_checks.py
│   ├── td.py
│   └── train_plantar_model.py
├── requirements.txt
└── README.md
```

Les dossiers de donnees `Events/` et `Plantar_activity_reference/` doivent etre places a la racine du projet, mais ils ne sont pas suivis par Git car ils sont volumineux.

## Installation

Depuis la racine du projet :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Dans VS Code ou Jupyter, selectionner ensuite le kernel Python du dossier `.venv`.

## Donnees attendues

Place les donnees comme ceci :

```text
.
├── Events/
│   └── S01/Sequence_01/classif.csv
└── Plantar_activity_reference/
    └── S01/Sequence_01/insoles.csv
```

Voir [data/README.md](data/README.md) pour le detail.

## Notebook de cours

Ouvrir :

```text
notebooks/cours_train_plantar_model.ipynb
```

Ce notebook explique :

- comment lire les fichiers
- comment aligner les signaux avec les labels
- comment construire les features
- comment faire un split `Train / Validation / Test`
- ce que signifient `epoch`, `batch_size` et `iteration`
- comment detecter l'overfitting

## Entrainement rapide

Pour verifier que tout fonctionne sans attendre longtemps :

```bash
.venv/bin/python scripts/train_plantar_model.py --max-files 8 --epochs 4
```

## Entrainement complet

```bash
.venv/bin/python scripts/train_plantar_model.py
```

Cette commande correspond a la meilleure configuration que nous avons gardee pour le TD :

```text
mode              = event
split             = random Train / Validation / Test
train / val / test= 70% / 15% / 15%
model             = MLPClassifier
hidden_layers     = 256,128
epochs            = 80
batch_size        = 64
learning_rate     = 0.001
alpha             = 0.0001
patience          = 12
random_state      = 42
```

Commande equivalente explicite :

```bash
.venv/bin/python scripts/train_plantar_model.py \
  --mode event \
  --split random \
  --model mlp \
  --hidden-layers 256,128 \
  --epochs 80 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --alpha 0.0001 \
  --patience 12 \
  --val-size 0.15 \
  --test-size 0.15 \
  --random-state 42
```

Le script genere dans `outputs/` :

- `plantar_activity_model.joblib`
- `plantar_activity_model_history.csv`
- `plantar_activity_model_metrics.json`
- `plantar_activity_model_test_predictions.csv`

Resultat observe avec cette configuration :

```text
samples              = 10 204
features             = 852
classes              = 31
train / val / test   = 7 142 / 1 531 / 1 531
best_epoch           = 21
best_val_accuracy    = 0.788
test_accuracy        = 0.781
balanced_accuracy    = 0.781
```

Le diagnostic signale un risque d'overfitting : la train accuracy monte tres haut (`0.999`) alors que la validation plafonne autour de `0.788`. L'early stopping garde donc le meilleur epoch validation, puis le test final reste coherent avec la validation (`0.781`).

Ce score est un resultat de split aleatoire unique, utile pour le TD et pour viser environ `80%`. Pour une evaluation plus exigeante, utiliser le split par sujet ci-dessous.

## Evaluation plus stricte

Pour tester la generalisation a des sujets jamais vus :

```bash
.venv/bin/python scripts/train_plantar_model.py --split subject --strict
```

Ce score est normalement plus difficile que le split aleatoire.

## Agreger les donnees pour exploration

Le notebook TD historique utilise des CSV agreges dans `outputs/`.

Pour les creer a partir de `Plantar_activity_reference/` et `Events/` :

```bash
.venv/bin/python scripts/aggregate_td_data.py
```

Puis verifier les donnees :

```bash
.venv/bin/python scripts/td_data_checks.py
```

## Regles importantes

- Ne pas committer `.venv/`.
- Ne pas committer les dossiers de donnees brutes.
- Ne pas choisir le modele final avec le test set.
- Utiliser la validation pendant l'entrainement, puis le test uniquement a la fin.

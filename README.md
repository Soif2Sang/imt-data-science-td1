# TD1 Plantar Activity Recognition

Projet pedagogique pour analyser des donnees de semelles instrumentees et entrainer un modele de reconnaissance d'activites.

Le repo contient :

- un script d'entrainement robuste avec protocole `Train / Validation / Test`, K-fold et ensembling
- un notebook de cours interactif pour comprendre le pipeline pas a pas
- un notebook de cours dedie au K-fold et a l'ensembling
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
│   ├── cours_kfold_ensembling_plantar_model.ipynb
│   ├── cours_train_plantar_model.ipynb
│   └── td_notebook.ipynb
├── scripts/
│   ├── aggregate_td_data.py
│   ├── td_data_checks.py
│   ├── td.py
│   ├── train_kfold_ensembling_plantar_model.py
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
notebooks/cours_kfold_ensembling_plantar_model.ipynb
```

Le premier notebook explique :

- comment lire les fichiers
- comment aligner les signaux avec les labels
- comment construire les features
- comment faire un split `Train / Validation / Test`
- ce que signifient `epoch`, `batch_size` et `iteration`
- comment detecter l'overfitting

Le second notebook explique :

- pourquoi on garde un test final separe
- comment fonctionne le K-fold
- pourquoi utiliser `StratifiedKFold` ou `GroupKFold`
- comment l'ensembling moyenne les probabilites
- comment lire les fichiers de sortie du script

## Entrainement rapide

Pour verifier que tout fonctionne sans attendre longtemps :

```bash
.venv/bin/python scripts/train_kfold_ensembling_plantar_model.py \
  --max-files 8 \
  --model extra_trees \
  --n-estimators 20 \
  --cv-folds 2 \
  --no-save
```

## Entrainement complet

```bash
.venv/bin/python scripts/train_kfold_ensembling_plantar_model.py \
  --model mlp \
  --cv-folds 5 \
  --epochs 80 \
  --batch-size 64 \
  --hidden-layers 256,128 \
  --learning-rate 0.001 \
  --alpha 0.0001 \
  --patience 12 \
  --output outputs/plantar_activity_mlp_cv_ensemble.joblib
```

Cette commande correspond a la meilleure configuration que nous avons gardee pour le TD. Elle ne part pas sur un modele custom : on garde `MLPClassifier` de scikit-learn, mais on ajoute une cross-validation K-fold et un ensembling final.

```text
mode                 = event
split                = random holdout test + StratifiedKFold
train+validation/test= 85% / 15%
cv_folds             = 5
ensemble             = moyenne des predict_proba des 5 modeles
model                = MLPClassifier
hidden_layers        = 256,128
epochs               = 80
batch_size           = 64
learning_rate        = 0.001
alpha                = 0.0001
patience             = 12
random_state         = 42
```

Commande equivalente explicite :

```bash
.venv/bin/python scripts/train_kfold_ensembling_plantar_model.py \
  --mode event \
  --split random \
  --model mlp \
  --cv-folds 5 \
  --hidden-layers 256,128 \
  --epochs 80 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --alpha 0.0001 \
  --patience 12 \
  --val-size 0.15 \
  --test-size 0.15 \
  --random-state 42 \
  --output outputs/plantar_activity_mlp_cv_ensemble.joblib
```

Le script genere dans `outputs/` :

- `plantar_activity_mlp_cv_ensemble.joblib`
- `plantar_activity_mlp_cv_ensemble_history.csv`
- `plantar_activity_mlp_cv_ensemble_metrics.json`
- `plantar_activity_mlp_cv_ensemble_test_predictions.csv`

Resultat observe avec cette configuration :

```text
samples                    = 10 204
features                   = 852
classes                    = 31
train+validation / test    = 8 673 / 1 531
fold_val_accuracy          = 0.757 / 0.775 / 0.775 / 0.779 / 0.782
ensemble_test_accuracy     = 0.813
ensemble_balanced_accuracy = 0.811
```

Le diagnostic signale encore un risque d'overfitting : la train accuracy de certains folds monte tres haut alors que la validation plafonne autour de `0.76-0.78`. L'early stopping garde donc le meilleur epoch validation pour chaque fold, puis l'ensemble des 5 modeles atteint `0.813` sur le test final.

Ce score depasse l'objectif d'environ `80%`, mais il reste base sur un holdout test aleatoire. Pour une evaluation plus exigeante, utiliser le split par sujet ci-dessous.

## Evaluation plus stricte

Pour tester la generalisation a des sujets jamais vus :

```bash
.venv/bin/python scripts/train_kfold_ensembling_plantar_model.py --split subject --strict --cv-folds 5
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

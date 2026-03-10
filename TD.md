# Organisation : Groupes de 4 étudiants
Objectif général : Découvrir et analyser les données plantaires avant de passer à la reconnaissance d’actions.

L’objectif du premier TD est de prendre en main le jeu de données et de comprendre ce que vous manipulez avant de passer à l’entraînement d’un modèle.

## Travaux à réaliser :

- Chargement des données

- Utiliser Python (numpy, éventuellement pandas) pour lire insoles.csv et classif.csv.

- Vérifier l’intégrité des fichiers et la cohérence des colonnes.

- Exploration et analyse des données

- Observer la distribution des valeurs pour chaque capteur (pression, IMU, force totale, centre de pression).

- Identifier les problèmes éventuels : valeurs aberrantes, données manquantes, anomalies dans les signaux.

- Comparer les variations entre :

	- différentes classes d’action

	- différents sujets

- Visualisation des signaux

- Proposer et tester des visualisations pour mieux comprendre les données :

	- Activation plantaire : cartes de pression ou courbes des 16 capteurs par pied (référez vous à l'image "position_capteurs" dans l'archive du TD.

	- IMU X/Y/Z : courbes des accélérations et vitesses angulaires

	- Centre de pression : trajectoire du centre de pression dans le plan X-Y

- Préparation pour le futur modèle

- Aligner les fenêtres de données avec les annotations (Timestamp Start/End)

- Noter vos observations sur les différences entre actions et sujets

- Réfléchir aux outils et transformations qui pourraient être utiles pour le prochain TD (normalisation, nettoyage, extraction de features, etc.)

## Objectif du TD

- Créer un notebook Python, comprenant :

	- Chargement des fichiers insoles.csv et classif.csv

	- Analyse statistique et visualisations

	- Identification et commentaire des anomalies et variations

	- Conclusions sur la structure et les caractéristiques des données

## Points clés pour ce TD

- L’objectif n’est pas encore d’entraîner un modèle, mais de comprendre le dataset.

- Pensez à documenter vos observations dans le notebook (texte + graphiques).

- Travail collaboratif en groupe de 4 : répartir les tâches (analyse, visualisation, documentation).

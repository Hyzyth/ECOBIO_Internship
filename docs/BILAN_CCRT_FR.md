# FlyScope — bilan pratique CCRT

Document de transmission pour l’équipe ECOBIO : dispositif de prise de vue, mesure du CCRT et reprise du développement. État du projet au 22 octobre 2026.

## Ce qui est utilisable aujourd’hui

FlyScope est une **application locale Python avec une interface dans le navigateur**. Ce n’est pas actuellement un exécutable `.exe`. Les images, annotations et modèles restent sur le PC ; aucun serveur distant ni abonnement n’est nécessaire. Un petit serveur local relie le navigateur au programme, à l’adresse `http://127.0.0.1:5000`.

L’application permet de choisir une séquence, définir son début utile et son origine temporelle, détecter/corriger les puits, confirmer les puits vides ou occupés, annoter Coma/Awake et reprendre un travail sauvegardé. Elle affiche les prédictions et permet de les corriger. Le réveil est considéré comme irréversible dans une séquence : une décision Awake confirmée se prolonge jusqu’à la fin. Les cas ambigus restent à vérifier. Le comptage d’œufs possède une page distincte ; son futur modèle reste à définir avec le format expérimental.

Trois approches sont entraînables et comparables sur les **mêmes individus et les mêmes partitions sauvegardées** : ResNet-18 (aspect), pose anatomique SLEAP + forêt aléatoire, et pipeline temporel DeepEthogram (mouvement, aspect puis contexte temporel). Jusqu’à dix sélections par approche : grille annotée complète hors validation, 25 individus, deux sélections de 10, deux de 5, deux de 2 et deux individus choisis pour leurs postures pauvres/riches. Les annotations partielles sont acceptées ; une absence d’annotation n’est jamais une classe négative.

## Résultats : distinguer outil fonctionnel et mesure automatique fiable

Le dernier jeu transmis comporte une grille d’évaluation de 61 individus, 1 313 images chacun, et **60 réveils observés**. Sur la sélection d’entraînement complète :

| Approche | Images correctement classées, avant filtrage temporel | Réveils détectés / 60 | Réveils à ±5 images / 60 |
| --- | ---: | ---: | ---: |
| ResNet-18 | 87,6 % | 58 | 29 (48,3 %) |
| DeepEthogram | 77,9 % | 0 | 0 |
| Pose + forêt | 58,9 % | 0 | 0 |

Pour ces deux derniers modèles, les contrôles de validation ont **désactivé la confirmation automatique**, faute de réglage suffisamment sûr. Cela signifie « vérification humaine nécessaire », pas « tous les individus restent dans le coma ». Certaines petites sélections fonctionnent différemment ; le tableau ne résume pas les trente variantes par une moyenne.

Pour ResNet, l’erreur moyenne de CCRT sur les 58 réveils détectés est de 85,7 images ; deux réveils sont manqués et 18 estimations sont trop précoces. Une ancienne méthode de calcul excluait des prédictions précédées d’une zone Unsure : elle donnait une moyenne artificiellement favorable. Les rapports corrigés comptent les événements manqués comme des échecs dans le taux à ±X images. **L’indicateur principal doit être la précision temporelle du CCRT, accompagnée du taux de détection**, et non la seule proportion d’images correctes. À une image/seconde, une erreur de cinq images correspond à cinq secondes ; cette conversion exige de connaître l’intervalle réel.

Ces résultats servent désormais au développement. Ils ne constituent pas une garantie sur une nouvelle expérience. Il faut une autre grille indépendante pour une évaluation finale après les corrections.

## Pourquoi les difficultés persistent

Les images Awake sont plus nombreuses, mais ce n’est pas l’unique problème. Le code pondère déjà les classes et sépare des individus pour la validation. Des milliers d’images voisines d’un seul individu restent très redondantes : une perte d’entraînement presque nulle peut traduire la mémorisation. Le meilleur point de contrôle est déjà choisi sur la validation, avec arrêt anticipé.

Le repérage anatomique appris sur le jeu public ne se transfère pas correctement aux vues CCRT. Même sur son jeu source, la tête et le thorax sont bien localisés, mais les extrémités des pattes le sont beaucoup moins. Une nouvelle page permet d’annoter directement les points anatomiques sur les drosophiles de nos puits, puis d’affiner le modèle. Les points cachés sont déclarés occultés, sans inventer leur position. Des individus complets sont réservés à la validation.

La confusion entre une drosophile comateuse sur le côté et une drosophile agrippée au bord est plausible, mais non démontrée par ces fichiers. Une vue 2D ne prouve pas le contact des pattes avec une paroi. Le modèle de pose actuel n’utilise pas explicitement la distance au bord. Il faut annoter des exemples représentatifs, y compris au bord des puits ; une pénalité arbitraire « ventre en l’air » n’a pas été ajoutée.

## Installation et lancement sur un PC

La procédure complète, avec les commandes d’entraînement, est dans [WORKFLOW.md](WORKFLOW.md). Une seule version de Python, un seul environnement et un seul `requirements.txt` sont utilisés. La configuration vérifiée est **Python 3.12.12 sous Linux**. L’installation native Windows et un éventuel paquet `.exe` restent à valider ; l’ancien projet Windows/Python 3.10 n’impose plus cette version au nouveau logiciel.

Sous Ubuntu/Debian, après installation de Python 3.12 :

```bash
sudo apt-get install -y libgl1 libglib2.0-0
cd ECOBIO_Internship
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip check
python tools/check_runtime.py --prepare-models
python app.py
```

Ouvrir ensuite `http://127.0.0.1:5000` dans un navigateur. Lors des utilisations suivantes, seules l’activation de l’environnement et la commande `python app.py` sont nécessaires. Garder le terminal ouvert et l’arrêter avec Ctrl+C après avoir sauvegardé les annotations. Les poids entraînés doivent être fournis séparément ; la présence du programme ne crée pas un modèle validé.

Pour charger un dossier de modèles entraînés, avant le lancement :

```bash
export FLYSCOPE_SUITE_MODELS_DIR="/chemin/vers/comparison-full"
python app.py
```

L’inférence utilise le CPU par défaut. Les commandes PowerShell correspondantes sont indiquées dans le workflow, à titre de procédure non encore validée sur Windows.

## Quelle puissance informatique ?

**Annoter et corriger :** PC classique avec navigateur et espace disque suffisant. Aucune carte NVIDIA n’est exigée. Les grandes séquences et exports restent coûteux en disque et en temps ; préférer un SSD. Le mode d’export direct dans un dossier, disponible sous Chromium/Chrome/Edge, évite de garder une grosse archive entière en mémoire.

**Utiliser un modèle entraîné :** CPU possible, testé fonctionnellement ; le traitement d’une séquence entière est plus lent qu’avec un GPU. Les temps sur le PC final et les minima de RAM restent à mesurer. Aucun engagement de temps réel ou de configuration mémoire minimale n’a encore été établi.

**Entraîner/affiner :** GPU NVIDIA recommandé pour les réseaux, mais la forêt aléatoire s’entraîne sur CPU. La RTX PRO 3000 Blackwell d’environ 12 Go est la cible prévue, avec Torch 2.8/CUDA 12.8 et un pilote adapté. Les vérifications du logiciel dans le cloud ont été faites sur CPU, faute de GPU physique. Le script `tools/check_runtime.py --require-cuda --prepare-models` vérifie les opérations sur la carte du laboratoire. La carte graphique est nécessaire surtout au poste de développement ; elle n’est pas imposée à chaque utilisateur final.

Le logiciel n’est pas encore un petit paquet autonome : les dépendances de recherche représentent plusieurs Go au téléchargement initial. Un export ONNX ou un exécutable avec un environnement réduit pourrait alléger une future version de production ; ces livrables ne sont pas fournis aujourd’hui.

## Comment partager et conserver le travail ?

Partager une version identifiée du dépôt, le workflow et un dossier de poids vérifiés. Recréer l’environnement sur le nouveau PC ; ne pas copier directement `.venv` entre systèmes. Pour travailler hors ligne, préparer à l’avance les dépendances, les poids et les caches des architectures. Les accès réseau servent à l’installation et aux téléchargements initiaux, pas à envoyer les images expérimentales à un service externe.

Trois exports ont des rôles distincts :

| Export | À quoi sert-il ? |
| --- | --- |
| Rapport lisible | CCRT par individu, état d’occupation, premier réveil, statut et carte de localisation des puits. |
| Sauvegarde du projet / analyse machine | Reprendre les annotations avec les images d’origine ; garder géométrie, corrections, probabilités et provenance. |
| Jeu d’entraînement ou de test | Images et annotations exploitables par les modèles. Le test est allégé des outils d’entraînement. |

L’export CCRT peut conserver uniquement la zone de tous les puits, avec une marge de 32 pixels et les corrections de position. PNG est sans perte, JPEG est plus compact avec une petite perte explicite. Les coordonnées et empreintes des images d’origine restent traçables. La sauvegarde éditable garde les coordonnées d’origine ; conserver les images originales pour reprendre le travail.

Sauvegarder avant de fermer le navigateur. Pour une archive en plusieurs parties, télécharger et extraire **toutes** les parties dans un même dossier vide. Un export n’est terminé qu’à l’apparition des liens de téléchargement ou du message confirmant le dossier sauvegardé.

## Mesure du temps et dispositif de prise de vue

Par défaut, l’image 1 est le temps zéro. Pour une séquence brute incluant l’installation des animaux, choisir l’image à partir de laquelle l’analyse commence : les indices et le temps précédents restent comptés. Pour une séquence déjà coupée, renseigner le délai connu entre la fin du froid et la première image reçue. Si ce délai est inconnu, rapporter les identifiants d’image plutôt qu’un CCRT absolu trompeur.

Avec l’ingénieur, privilégier une caméra stable, une lumière homogène, les puits entièrement visibles et des bords nets. Vérifier que le contour conserve les pattes des individus proches du bord. La détection automatique est une proposition corrigible ; un puits absent ou masqué n’est pas silencieusement une absence d’individu. L’hypothèse actuelle est un individu par puits ; les cas multiples/masqués sont signalés et exclus des cibles comportementales usuelles.

## Prochaines décisions avec l’équipe

- Définir la tolérance acceptable sur le CCRT : ±X images/secondes, risque acceptable d’un réveil trop précoce et traitement des animaux déjà réveillés au début.
- Annoter l’anatomie d’individus CCRT représentatifs, puis vérifier les superpositions et les erreurs sur d’autres individus.
- Comparer un petit pilote avec caractéristiques fixes (`--backbone frozen`) au fine-tuning, uniquement sur la validation ; ne pas choisir sur la grille finale.
- Réserver de nouvelles grilles pour vérifier la robustesse à l’éclairage, aux supports et aux espèces.
- Mesurer le temps de traitement sur le PC utilisateur et définir ensuite le besoin d’un empaquetage simplifié.

Les graphes entraînement/validation sont maintenant séparés. Les trois courbes DeepEthogram correspondent à ses trois étapes ; les forêts n’ont pas d’époques, mais disposent de métriques de validation. Une perte de 0,03 n’est pas « 3 % d’erreur ». Les définitions et exemples chiffrés sont détaillés dans [SECOND_TRAINING_REVIEW.md](SECOND_TRAINING_REVIEW.md).

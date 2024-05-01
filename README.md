# Fine-tuning t5-small pour BPMN

Ce dépôt contient un script Python ainsi que les jeux de données nécessaires pour le fine-tuning du modèle T5, dans le but de générer des graphes en notation BPMN à partir de descriptions de processus. Le projet utilise PyTorch et la bibliothèque Transformers de Hugging Face.

## Prérequis
- Python 3.9
- Pip (gestionnaire de paquets pour Python)

## Installation

Commencez par cloner ce dépôt :

```bash
git clone https://github.com/ofachati/Finetuning-T5small-BPMN
```

installez les dépendances requises en exécutant :

```bash
pip install -r requirements.txt
```
#### Ces dépendances fonctionnent pour Windows, mais n'ont pas été testées pour d'autres systèmes d'exploitation.
## Exécution du script

Pour lancer le script de fine-tuning.
```bash
python finetune.py
```

Assurez-vous que les chemins vers les fichiers de données dans le script sont corrects et pointent vers les fichiers JSON appropriés dans le dossier `Dataset`.


## Détails du Code

### Sélection des Jeux de Données pour l'Entraînement

Le script est préconfiguré avec des sections de code commentées pour charger différents ensembles de données correspondant à 15 domaines de processus d'affaires. Pour démarrer l'entraînement sur un jeu de données spécifique, il suffit de décommenter les lignes appropriées et de commenter les autres :

```python
# Processus de paiement des comptes
train_data = load_and_format_data('Dataset/train/account_payable_process.json')
validation_data = load_and_format_data('Dataset/validation/account_payable_process.json')

# Processus de recouvrement des comptes
# train_data = load_and_format_data('Dataset/train/accounts_receivable_process.json')
# validation_data = load_and_format_data('Dataset/validation/accounts_receivable_process.json')
```

Pour une preuve de concept, il suffit d'entraîner seulement sur 2 ou 3 datasets afin de développer 2 ou 3 modèles experts dans ces domaines spécifiques.

### Gestion de l'Entraînement et Reprise à Partir d'un Point de Sauvegarde

Le script permet de reprendre l'entraînement à partir d'un checkpoint spécifique si nécessaire. Nous utilisons principalement un checkpoint si nous arrêtons l'entraînement au milieu d'un jeu de données spécifique. Lorsque nous commençons avec un nouveau jeu de données, nous utilisons `trainer.train()`.

Pour reprendre l'entraînement à partir d'un checkpoint, décommentez et modifiez la ligne suivante selon le checkpoint désiré :

```python
# Utiliser ceci à la place si vous voulez reprendre l'entraînement à partir d'un checkpoint
trainer.train(resume_from_checkpoint="./results/checkpoint-1000")
```

Remplacez `"./results/checkpoint-1000"` par le chemin vers votre fichier de checkpoint.

Si vous ne souhaitez pas reprendre à partir d'un checkpoint, utilisez simplement :

```python
trainer.train()
```

Cela démarrera un nouveau cycle d'entraînement à partir de zéro.

**Remarque importante :** Après avoir terminé l'entraînement sur un Dataset, assurez-vous de sauvegarder les fichiers du modèle ou de les renommer pour éviter qu'ils ne soient écrasés par un nouveau modèle entraîné.



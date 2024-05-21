from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json

# charger et formater les données
def load_and_format_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    formatted_data = []
    for item in data:
        formatted_data.append((item['txt1'], item['dot']))
        formatted_data.append((item['txt2'], item['dot']))
    return formatted_data

# Charger les données d'entraînement et de validation pour chaque type de processus

# Processus de Gestion de Projets et Processus
train_data = load_and_format_data('Dataset/train/gestion_projets_et_processus.json')
validation_data = load_and_format_data('Dataset/validation/gestion_projets_et_processus.json')

# Processus de Gestion Financière
# train_data = load_and_format_data('Dataset/train/gestion_financiere.json')
# validation_data = load_and_format_data('Dataset/validation/gestion_financiere.json')

# Processus de Gestion Clientèle et Marketing
# train_data = load_and_format_data('Dataset/train/gestion_clientele_et_marketing.json')
# validation_data = load_and_format_data('Dataset/validation/gestion_clientele_et_marketing.json')

# Processus de Gestion de Projets et Processus
# train_data = load_and_format_data('Dataset/train/gestion_projets_et_processus.json')
# validation_data = load_and_format_data('Dataset/validation/gestion_projets_et_processus.json')

# Processus de Gestion des Ressources Humaines
# train_data = load_and_format_data('Dataset/train/gestion_ressources_humaines.json')
# validation_data = load_and_format_data('Dataset/validation/gestion_ressources_humaines.json')

# Processus de Gestion de Startups et Innovation
# train_data = load_and_format_data('Dataset/train/gestion_startups_innovation.json')
# validation_data = load_and_format_data('Dataset/validation/gestion_startups_innovation.json')

# Charger tokenizer et modèle
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Classe Dataset personnalisée
class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=1024):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer.encode_plus(item[0], max_length=self.max_length, padding='max_length',
                                                    truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer.encode_plus(item[1], max_length=self.max_length, padding='max_length',
                                                     truncation=True, return_tensors="pt")
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# Préparer les datasets
train_dataset = CustomDataset(tokenizer, train_data)
validation_dataset = CustomDataset(tokenizer, validation_data)

# Définir les arguments de l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_steps=1000,  # Sauvegarder un checkpoint tous les 1000 pas
    save_total_limit=3,  # Garder seulement les 3 derniers checkpoints
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle affiné
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

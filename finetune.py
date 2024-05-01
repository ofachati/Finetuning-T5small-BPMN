from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json

# Fonction pour charger les données
def load_and_format_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    formatted_data = []
    for item in data:
        # Ajouter txt1 et txt2 avec la même représentation dot
        formatted_data.append((item['txt1'], item['dot']))
        formatted_data.append((item['txt2'], item['dot']))
    return formatted_data

# Charger les données d'entraînement et de validation
train_data = load_and_format_data('Dataset/train/account_payable_process.json')
validation_data = load_and_format_data('Dataset/validation/account_payable_process.json')

# Charger les données d'entraînement et de validation pour chaque type de processus

# Account Payable Process
train_data = load_and_format_data('Dataset/train/account_payable_process.json')
validation_data = load_and_format_data('Dataset/validation/account_payable_process.json')

# Accounts Receivable Process
# train_data = load_and_format_data('Dataset/train/accounts_receivable_process.json')
# validation_data = load_and_format_data('Dataset/validation/accounts_receivable_process.json')

# Budget Preparation Process
# train_data = load_and_format_data('Dataset/train/budget_preparation_process.json')
# validation_data = load_and_format_data('Dataset/validation/budget_preparation_process.json')

# Churn Rate Prevention Process
# train_data = load_and_format_data('Dataset/train/churn_rate_prevention_process.json')
# validation_data = load_and_format_data('Dataset/validation/churn_rate_prevention_process.json')

# Client Onboarding Process for a Marketing Agency
# train_data = load_and_format_data('Dataset/train/client_onboarding_process_for_a_marketing_agency.json')
# validation_data = load_and_format_data('Dataset/validation/client_onboarding_process_for_a_marketing_agency.json')

# Content Promotion Process
# train_data = load_and_format_data('Dataset/train/content_promotion_process.json')
# validation_data = load_and_format_data('Dataset/validation/content_promotion_process.json')

# Customer Support Process for the Ticket Management
# train_data = load_and_format_data('Dataset/train/customer_support_process_for_the_ticket_management.json')
# validation_data = load_and_format_data('Dataset/validation/customer_support_process_for_the_ticket_management.json')

# Employee Onboarding Process
# train_data = load_and_format_data('Dataset/train/employee_onboarding_process.json')
# validation_data = load_and_format_data('Dataset/validation/employee_onboarding_process.json')

# Final Grades Submission Process
# train_data = load_and_format_data('Dataset/train/final_grades_submission_process.json')
# validation_data = load_and_format_data('Dataset/validation/final_grades_submission_process.json')

# Loan Application Process
# train_data = load_and_format_data('Dataset/train/loan_application_process.json')
# validation_data = load_and_format_data('Dataset/validation/loan_application_process.json')

# Order Fulfillment Process
# train_data = load_and_format_data('Dataset/train/order_fulfillment_process.json')
# validation_data = load_and_format_data('Dataset/validation/order_fulfillment_process.json')

# Process for Optimizing a Process
# train_data = load_and_format_data('Dataset/train/process_for_optimizing_a_process.json')
# validation_data = load_and_format_data('Dataset/validation/process_for_optimizing_a_process.json')

# Project Management Process
# train_data = load_and_format_data('Dataset/train/project_management_process.json')
# validation_data = load_and_format_data('Dataset/validation/project_management_process.json')

# Purchase Order Workflow
# train_data = load_and_format_data('Dataset/train/purchase_order_workflow.json')
# validation_data = load_and_format_data('Dataset/validation/purchase_order_workflow.json')

# Startup Due Diligence for a Venture Capitalist
# train_data = load_and_format_data('Dataset/train/startup_due_diligence_for_a_venture_capitalist.json')
# validation_data = load_and_format_data('Dataset/validation/startup_due_diligence_for_a_venture_capitalist.json')




# Charger  tokenizer eet modele
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
# utiliser ceci à la place si vous voulez reprendre l'entraînement à partir d'un checkpoint
# trainer.train(resume_from_checkpoint="./results/checkpoint-34000")

# Sauvegarder le modèle affiné
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

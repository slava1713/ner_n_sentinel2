import json
import string
import pandas as pd
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch import cuda
from torch.utils.data import DataLoader


from model import Model
from config import Config
from dataset import Dataset


# Check if CUDA is available, otherwise use CPU
device = 'cuda' if cuda.is_available() else 'cpu'

def process_dataframe(df):
    # Group the DataFrame by 'sentence_id' and aggregate the words and labels into lists
    grouped_data = df.groupby('sentence_id').agg({'words': list, 'labels': list})

    # Create a list to store full sentences and a list of lists for labels
    sentences = []
    labels_list = []

    # Iterate through the grouped data and combine words into sentences and labels into lists
    for _, row in grouped_data.iterrows():
        # Encode labels using LabelEncoder
        encoded_label = le.fit_transform(row['labels'])
        
        sentences.append(row['words'])
        labels_list.append(encoded_label)

    # Create the DataFrame after the loop is complete
    correct_df = pd.DataFrame({'sentences': sentences, 'labels': labels_list})
        
    return correct_df

# Load dataset from CSV
df = pd.read_csv('data\dataset.csv')

# Encode labels using LabelEncoder
le = LabelEncoder().fit(df['labels'])
df['labels'] = le.transform(df['labels'])

# Group by sentence_id and aggregate words and labels into lists
data_gr = df.groupby("sentence_id").agg({'words': list, 'labels': list}).reset_index()

# Losses for demo.ipynb
train_losses = []
val_losses = []

# Train Function
def train_fn(train_data_loader, model, optimizer, device, scheduler):
    # Train the Model
    model.train()
    loss_ = 0
    for data in tqdm(train_data_loader, total=len(train_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)

        # Backward Propagation
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ += loss.item()
        
    train_losses.append(loss_ / len(train_data_loader))
    
    return model, loss_ / len(train_data_loader)

# Val Function
def val_fn(val_data_loader, model, device):
    model.eval()
    loss_ = 0
    
    for data in tqdm(val_data_loader, total=len(val_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)
        _, loss = model(**data)
        loss_ += loss.item()
        
    val_losses.append(loss_ / len(val_data_loader))
        
    return loss_ / len(val_data_loader)

# Split data into training and validation sets
train_sent, val_sent, train_tag, val_tag = train_test_split(data_gr['words'], data_gr['labels'], test_size=0.2, random_state=10, shuffle=False)
val_sent = val_sent.reset_index(drop=True)
val_tag = val_tag.reset_index(drop=True)

# Create datasets and data loaders
train_dataset = Dataset(texts=train_sent, tags=train_tag, tokenizer=Config.TOKENIZER)
val_dataset = Dataset(texts=val_sent, tags=val_tag, tokenizer=Config.TOKENIZER)
train_data_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE)
val_data_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE)

# Number of unique labels in the dataset
num_tag = len(df.labels.value_counts())

# Initialize the model
model = Model(num_tag=num_tag)
model.to(device)

def get_hyperparameters(model, full_finetuning):
    # ff: full_finetuning
    if full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters

# Set FULL_FINETUNING to True to fine-tune all layers of the model
FULL_FINETUNING = True

# Get optimizer hyperparameters based on FULL_FINETUNING
optimizer_grouped_parameters = get_hyperparameters(model, FULL_FINETUNING)

# Initialize AdamW optimizer with specified learning rate
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)

# Calculate the total number of training steps for the specified number of epochs
num_train_steps = int(len(train_sent) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)

# Set up a linear learning rate scheduler with warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_train_steps
)

# Initialize the best loss and epoch variable
best_loss = 100
best_epoch = 0

# Path to save model weights 
weights_path = Config.PATH

# Training loop
for epoch in range(Config.EPOCHS):
    # Train the model on the training data
    model, train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)

    # Validate the model on the validation data
    val_loss = val_fn(val_data_loader, model, device)
    
    print(f"Epoch: {epoch + 1}, Train_loss: {train_loss}, Val_loss: {val_loss}")
    
    # Update the best loss and epoch and save the model if it's improved
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1
        print(f"Model Updated Best Loss: {best_loss}   Epoch: {best_epoch}")
        torch.save(model.state_dict(), weights_path)
        
    
data = {"train_losses": train_losses, "val_losses": val_losses, "best_loss": best_loss, "best_epoch": best_epoch}

# Specify the file path where you want to save the JSON file
file_path = "demo.json"

# Save the data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(data, json_file)    

print(f"Training ended best loss: {best_loss}   Epoch: {best_epoch}")
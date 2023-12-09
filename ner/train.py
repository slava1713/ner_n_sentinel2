import pandas as pd
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import string
from torch import cuda

from model import Model
from config import Config
from dataset import Dataset
 


device = 'cuda' if cuda.is_available() else 'cpu'
 
def process_dataframe(df):
    
    # Group the DataFrame by 'sentence_id' and aggregate the words and labels into lists
    grouped_data = df.groupby('sentence_id').agg({'words': list, 'labels': list})

    # Create a list to store full sentences and a list of lists for labels
    sentences = []
    labels_list = []

    # Iterate through the grouped data and combine words into sentences and labels into lists
    for _, row in grouped_data.iterrows():
        encoded_label = le.fit_transform(row['labels'])
        
        sentences.append(row['words'])
        labels_list.append(encoded_label)

    # Create the DataFrame after the loop is complete
    correct_df = pd.DataFrame({'sentences': sentences, 'labels': labels_list})
        
    return correct_df

df = pd.read_csv('dataset.csv')


le = LabelEncoder().fit(df['labels'])
df['labels'] = le.transform(df['labels'])

data_gr = df.groupby("sentence_id").agg({'words': list, 'labels': list}).reset_index()

def train_fn(train_data_loader, model, optimizer, device, scheduler):
    #Train the Model
    model.train()
    loss_ = 0
    for data in tqdm(train_data_loader, total = len(train_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)

        #Backward Propagation
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ += loss.item()
    
    return model, loss_ / len(train_data_loader)


def val_fn(val_data_loader, model, optimizer, device, scheduler, best_loss):
    model.eval()
    loss_ = 0
    
    for data in tqdm(val_data_loader, total = len(val_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)
        _, loss = model(**data)
        loss_ += loss.item()
        
    return loss_ / len(val_data_loader)


train_sent, val_sent, train_tag, val_tag = train_test_split(data_gr['words'], data_gr['labels'], test_size=0.2, random_state=10, shuffle=False)
val_sent = val_sent.reset_index(drop=True)
val_tag = val_tag.reset_index(drop=True)

train_dataset = Dataset(texts=train_sent, tags=train_tag, tokenizer=Config.TOKENIZER)
val_dataset = Dataset(texts=val_sent, tags=val_tag, tokenizer=Config.TOKENIZER)
train_data_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE)
val_data_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE)

num_tag = len(df.labels.value_counts())
model = Model(num_tag=num_tag)
model.to(device)

def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
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

FULL_FINETUNING = True
optimizer_grouped_parameters = get_hyperparameters(model, FULL_FINETUNING)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)
num_train_steps = int(len(train_sent) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)

scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_train_steps
)

best_loss = 100

for epoch in range(Config.EPOCHS):
    model, train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
    val_loss = val_fn(val_data_loader, model, optimizer, device, scheduler, best_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss
        print(f"Model Updated Best Loss: {best_loss}")
        
        torch.save(model.state_dict(), Config.PATH)
        
    print(f"Epoch: {epoch + 1}, Train_loss: {train_loss}, Val_loss: {val_loss}")
    
    
def prediction(test_sentence, model, le):
    for i in list(string.punctuation):
        test_sentence = test_sentence.replace(i, ' ' + i)
    test_sentence = test_sentence.split()
    print(test_sentence)
    Token_inputs = Config.TOKENIZER.encode(test_sentence, add_special_tokens=False)
    print(Token_inputs)
    test_dataset =  Dataset(test_sentence, tags= [[1] * len(test_sentence)], tokenizer=Config.TOKENIZER)
    num_tag = len(le.classes_)
   
    with torch.no_grad():
        data = test_dataset[0]
        for i, j in data.items():
            data[i] = j.to(device).unsqueeze(0)
        tag, _ = model(**data)

        print(le.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(Token_inputs)+1])
        
test_sentences = [
    "Makovytsia, with its accessible trails, is suitable for hikers of all skill levels.",
    "Petros' summit, kissed by the first light of dawn, is a magical moment for early risers.",
    "Shpici, hidden away from the tourist crowds, provides a tranquil escape for those seeking solitude.",
    "The valleys surrounding Hoverla are home to unique plant species adapted to the mountainous climate.",
    "Sinyak's craggy peaks are silhouetted against the evening sky, creating a dramatic scene.",
    "Makovytsia's diverse ecosystem supports a variety of wildlife, from butterflies to elusive mammals.",
    "Petros, with its challenging ascent, attracts climbers from around the world.",
    "Shpici's remote location allows for stargazing, offering a clear view of the night sky."
    ]

for test_sentence in test_sentences:
    prediction(test_sentence, model, le)
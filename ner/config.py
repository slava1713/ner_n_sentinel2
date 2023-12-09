from transformers import BertTokenizer

class Config:
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 8
    EPOCHS = 20
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    PATH = "model.pth"
    
    CLS = [101]
    SEP = [102]
    VALUE_TOKEN = [0]
    MAX_LEN = 32
import torch
from torch.utils.data import Dataset
from config import Config
        
    
class Dataset:
  
  def __init__(self, texts, tags, tokenizer):
    
    self.tokenizer = tokenizer
    self.texts = texts
    self.tags = tags
  
  def __len__(self):
    return len(self.texts)

  def __getitem__(self, index):
    
    texts = self.texts[index]
    tags = self.tags[index]
    
    #Tokenise
    ids = []
    target_tag = []

    for i, s in enumerate(texts):
        inputs = self.tokenizer.encode(s, add_special_tokens=False)
     
        input_len = len(inputs)
        ids.extend(inputs)
        target_tag.extend(input_len * [tags[i]])
    
    #To Add Special Tokens, subtract 2 from MAX_LEN
    ids = ids[:Config.MAX_LEN - 2]
    target_tag = target_tag[:Config.MAX_LEN - 2]

    #Add Sepcial Tokens
    ids = Config.CLS + ids + Config.SEP
    target_tags = Config.VALUE_TOKEN + target_tag + Config.VALUE_TOKEN

    mask = [1] * len(ids)
    token_type_ids = [0] * len(ids)

    #Add Padding if the input_len is small

    padding_len = Config.MAX_LEN - len(ids)
    ids = ids + ([0] * padding_len)
    target_tags = target_tags + ([0] * padding_len)
    mask = mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)

    return {
        "ids" : torch.tensor(ids, dtype=torch.long),
        "mask" : torch.tensor(mask, dtype=torch.long),
        "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
        "target_tags" : torch.tensor(target_tags, dtype=torch.long)
      }

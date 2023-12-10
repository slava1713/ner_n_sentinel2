import torch
import transformers


class Model(torch.nn.Module):
    
    def __init__(self, num_tag):
        super(Model, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = torch.nn.Dropout(0.3)
        self.out_tag = torch.nn.Linear(768, self.num_tag)
        
    #Forward Pass
    def forward(self, ids, mask, token_type_ids, target_tags):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bert_out = self.bert_drop(output) 
        tag = self.out_tag(bert_out)
    
        #Calculate the loss
        Critirion_Loss = torch.nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = tag.view(-1, self.num_tag)
        active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(Critirion_Loss.ignore_index).type_as(target_tags))
        loss = Critirion_Loss(active_logits, active_labels)
        return tag, loss

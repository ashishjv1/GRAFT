from transformers import AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F


class bertmodel(nn.Module):
    def __init__(self, device, numlabels=2):
        super(bertmodel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=numlabels)
        self.device = device
            
    def forward(self, x, indices, last=False, freeze=False):
        if freeze and last and not indices:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            
            output = self.model(x["input_ids"].to(self.device), 
                        attention_mask=x["attention_mask"].to(self.device), labels=x["label"].to(self.device))
        elif freeze and last and indices:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            
            output = self.model(x["input_ids"][indices].to(self.device), 
                        attention_mask=x["attention_mask"][indices].to(self.device), labels=x["label"][indices].to(self.device))
            
        else:
#             for param in self.model.parameters():
#                 param.requires_grad = True
            output = self.model(x["input_ids"].to(self.device), 
                        attention_mask=x["attention_mask"].to(self.device), labels=x["label"].to(self.device))
         
            
        return output
        
   
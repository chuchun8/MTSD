import torch
import torch.nn as nn
from transformers import AutoModel, BertModel


# BERT/BERTweet
class stance_classifier(nn.Module):

    def __init__(self,num_labels,model_select,dropout):

        super(stance_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        if model_select == 'Bertweet':
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        elif model_select == 'Bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear2 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 2)
        
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len, x_input_ids2):
        
        last_hidden = self.bert(input_ids=x_input_ids, \
                                attention_mask=x_atten_masks, token_type_ids=x_seg_ids, \
                               )
        last_hidden2 = self.bert(input_ids=x_input_ids2, \
                                attention_mask=x_atten_masks, token_type_ids=x_seg_ids, \
                               )
        
        query = last_hidden[0][:,0]
        query2 = last_hidden2[0][:,0]
        query = self.dropout(query)
        query2 = self.dropout(query2)
        context_vec = torch.cat((query, query2), dim=1)
        
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        linear2 = self.relu(self.linear2(context_vec))
        out2 = self.out2(linear2)
        
        return out, out2

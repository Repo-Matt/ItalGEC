
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM
from encoder_sentence import tokenizer
"""# ***---> BART+BERT:***"""
model_huggingface="google/mt5-base"
model_huggingface="facebook/mbart-large-50"
device="cuda"
class MLP_arg_classification(nn.Module):
    def __init__(self, ):
        super(MLP_arg_classification, self).__init__()
        self.automodel = AutoModelForSeq2SeqLM.from_pretrained(model_huggingface)
        self.automodel.model.shared.weight.requires_grad=False
        #self.automodel =  T5ForConditionalGeneration.from_pretrained("t5-base")

    def forward(self, x,mask,correct):
      hidden_states = self.automodel(input_ids=x,attention_mask=mask,labels=correct)
      return hidden_states
    def generate(self, x,mask,max_length,num_beams):
      hidden_states = self.automodel.generate(x,attention_mask=mask,
                                              forced_bos_token_id=tokenizer.lang_code_to_id["it_IT"],
                                              max_length=max_length,num_beams=4)
      
      #hidden_states = self.automodel.generate(x,attention_mask=mask,max_length=max_length,num_beams=4)
      return hidden_states
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
############################################################################################################################################
#tokenizer =  BartTokenizer.from_pretrained('facebook/bart-base')
#tokenizer = AutoTokenizer.from_pretrained(model_huggingface)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="it_IT", tgt_lang="it_IT")
#tokenizer =  T5Tokenizer.from_pretrained('t5-base')


def encode_sentence(data1,data2):
  tokenized_seq = []
  labels=[]
  tokens_res=[]
  tokenized_corr=[]
  max_length=80
  
  print("tokenized_seq")
  tokenized_seq=tokenizer(data1, return_tensors='pt', padding=True, truncation=True,max_length=80)
  print("tokenized_corr")
  tokenized_corr=tokenizer(data2, return_tensors='pt', padding=True, truncation=True,max_length=80)
  
  tokenized_seq['input_ids']=tokenized_seq['input_ids'].type(torch.int64)
  tokenized_corr['input_ids']=tokenized_corr['input_ids'].type(torch.int64)
  tokenized_seq['attention_mask']=tokenized_seq['attention_mask'].type(torch.int64)
  tokenized_corr['attention_mask']=tokenized_corr['attention_mask'].type(torch.int64)

  print(tokenized_seq['input_ids'].shape,tokenized_corr['input_ids'].shape,tokenized_seq['attention_mask'].shape)
  """
  Unhused version for manual modification of sentences
    if tokenized_seq['input_ids'].shape[-1]<max_length:
    tokenized_seq['input_ids']=torch.cat((tokenized_seq['input_ids'],torch.ones([tokenized_seq['input_ids'].shape[0],max_length-tokenized_seq['input_ids'].shape[-1]])),-1)
  
  if tokenized_seq['attention_mask'].shape[-1]<max_length:
    tokenized_seq['attention_mask']=torch.cat((tokenized_seq['attention_mask'],torch.ones([tokenized_seq['attention_mask'].shape[0],max_length-tokenized_seq['attention_mask'].shape[-1]])),-1)

  if tokenized_corr['input_ids'].shape[-1]<max_length:
    tokenized_corr['input_ids']=torch.cat((tokenized_corr['input_ids'],torch.ones([tokenized_corr['input_ids'].shape[0],max_length-tokenized_corr['input_ids'].shape[-1]])),-1)
  
  if tokenized_corr['attention_mask'].shape[-1]<max_length:
    tokenized_corr['attention_mask']=torch.cat((tokenized_corr['attention_mask'],torch.ones([tokenized_corr['attention_mask'].shape[0],max_length-tokenized_corr['attention_mask'].shape[-1]])),-1)
  """
  
  return tokenized_seq,tokenized_corr


def encode_sentence_Batch(data1,data2):
  tokenized_seq = {"input_ids":[],"attention_mask":[]}
  tokenized_corr= {"input_ids":[],"attention_mask":[]}
  
  for elem1,elem2 in zip(data1,data2):
    tokenized_seq["input_ids"].append(tokenizer(elem1, return_tensors='pt')["input_ids"])
    tokenized_seq["attention_mask"].append(tokenizer(elem1, return_tensors='pt')["attention_mask"])
    tokenized_corr["input_ids"].append(tokenizer(elem2, return_tensors='pt')["input_ids"])
    tokenized_corr["attention_mask"].append(tokenizer(elem2, return_tensors='pt')["attention_mask"])


  #print(tokenized_seq['input_ids'][12].shape,tokenized_corr['input_ids'][12].shape,tokenized_seq['attention_mask'][12].shape)
  #print(tokenized_seq['input_ids'][44].shape,tokenized_corr['input_ids'][44].shape,tokenized_seq['attention_mask'][44].shape)
  
  return tokenized_seq,tokenized_corr
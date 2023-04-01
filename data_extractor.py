import torch
from typing import Dict, Optional, List, Any, Tuple
import spacy
nlp = spacy.load('it_core_news_sm')
max_len=10
classes_ERRANT={}
############################################################################################################################################
class ReaderM2(torch.utils.data.IterableDataset):
    def __init__(self, txt_path1,txt_path2,vocab_label):
        ##====================reading data====================
        self.vocab_label=vocab_label
        self.max_len=10
        self.nlp = nlp
        print("Getting sentences...")
        self.X_list, self.Y_list, self.X_corr = self.jsonlnp(txt_path1,txt_path2)
        
    def read_labels(self,lbl):
      count = {}
      add=0
      for scan in lbl:
        for s in scan:
          if s not in count:
            count[s]=add
            add+=1
      return count

    def jsonlnp(self,path: str,path_corr: str) -> Tuple[List[Dict], List[str]]:
      
        elem = []
        labels = []

        with open(path , encoding="utf-8") as f:
          data = f.readlines()
          app_labels=[]
          for obj in data:
            app=[]

            #sentence
            if obj[0]=="S":
              #doc = self.nlp(obj[2:-1])
              doc = obj[2:-1]
              elem.append(str(doc))

            #label
            elif obj[0]=="A":
              #whole info A
              app_strings=obj[:-1].split("|||")[0].replace(" ","|||")+"|||"+"|||".join(obj[:-1].split("|||")[1:])
              app_sent=app_strings.split("|||")
              #errant classes
              if app_sent[3] not in classes_ERRANT:
                classes_ERRANT[app_sent[3]]=len(classes_ERRANT)
              app_labels.append(app_sent)
              # space
            else:
              labels.append(app_labels)
              app_labels=[]

        #correct sentence
        correct_sentence=[]
        with open(path_corr , encoding="utf-8") as f:
          data = f.readlines()
          app_labels=[]
          for obj in data:
            app=[]
            #doc = self.nlp(obj[:-1])
            doc = obj[:-1]
            #for token in doc:
              #app.append(token.text)
            correct_sentence.append(str(doc))
            #correct_sentence.append(app)

        print(len(correct_sentence))
        return elem, labels, correct_sentence
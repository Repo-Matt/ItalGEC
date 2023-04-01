import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import wandb
import spacy
from pytorch_lightning.loggers import WandbLogger
from data_extractor import *
import errant_env.errant.errant as errant
annotator = errant.load('it',nlp)
device="cuda"
"""# ***---> Lighnting BART***"""
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="it_IT", tgt_lang="it_IT")
class Model_Correction(pl.LightningModule):
    def __init__(self,test,model_main,verbose_m2, embeddings = None, *args, **kwargs):
        super(Model_Correction, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        #WandB
        wandb.init(project='baseline-bart-base', name='MBart-batch-Step0.5')
        self.wandb_logger = WandbLogger()
        #Train parameters
        self.model = model_main
        wandb.config.update(model_main.automodel.config.to_dict())
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
        #Parameters
        self.tot_loss = []
        self.my_predictions= []
        self.my_labels= []
        self.my_inputs= []
        self.dataset_written=False
        self.validation_predictions = []
        self.testing_predictions= []
        self.max_length=80
        self.test=test
        self.outputs=[]
        self.lr = 1e-5
        self.scheduler=None
        self.verbose_m2=verbose_m2
        self.accumulate_lr1=1
        self.accumulate_lr2=1
        self.f05_monitor=0
        self.accumulate_lr3=1
        # print the current memory usage again
    
    def compute_loss(self, logits,mask, batch,mask_lbl): 
      #hidden_states = self.model.forward(torch.cat((logits,torch.ones(logits.shape[0],
      #                                                                  self.max_length-logits.shape[-1]).to(device)),-1).type(torch.int64),mask,batch)
      hidden_states = self.model.forward(logits.type(torch.int64),mask,batch)
      return hidden_states.logits,hidden_states.loss

    def forward(self, inputs,mask,labels,mask_lbl): 
      logits, loss=self.compute_loss(inputs,mask,labels,mask_lbl)
      return logits, loss
      
    def training_step(self, batch, batch_nb):
        inputs = batch[0]
        labels = batch[1]
        mask = batch[2]
        mask_lbl = batch[3]
        
        logits  ,loss = self.forward(inputs,mask,labels,mask_lbl)
        wandb.log({"train_loss": loss,"batch": batch_nb,"epoch": self.current_epoch})
        self.log('train_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def evaluation(self, inputs,mask,labels,mask_lbl):
        output = self.model.generate(inputs,mask,self.max_length,4)
        logits, loss =self.compute_loss(inputs,mask,labels,mask_lbl)
        #perplexity = torch.exp(self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1)).mean()).item()
        #wandb.log({"perplexity": perplexity,"epoch": self.current_epoch})
        if self.test:
          self.testing_predictions.append(output)
        else:
          self.validation_predictions.append(output)
        return loss
    
    def validation_step(self, batch, batch_nb):
        inputs = batch[0]
        labels = batch[1]
        mask = batch[2]
        mask_lbl = batch[3]

        loss = self.evaluation(inputs,mask,labels,mask_lbl)
        wandb.log({"valid_loss": loss,"batch": batch_nb})
        self.log('valid_loss', loss, prog_bar=True)
        self.outputs.append((labels,inputs))
        #return (labels,inputs)
 
    def test_step(self, batch, batch_nb):
        inputs = batch[0]
        labels = batch[1]
        mask = batch[2]
        mask_lbl = batch[3]
        
        output = self.model.generate(inputs,mask,self.max_length,4)
        loss = self.evaluation(inputs,mask,labels,mask_lbl)
        return {'log': inputs}
    
    def l_lambda1(self,epoch):
        epoch=self.current_epoch
        if epoch==5:
           self.accumulate_lr1=0.25

        if epoch==7:
           self.accumulate_lr1=0.125

        if epoch==2:
           self.accumulate_lr1=0.5
        return self.accumulate_lr1
    
    def get_F05(self,inputs,labels,predictions):
        TP=1
        FP=1
        FN=1

        ########### TYPES ###########
        # FP/TP/FN

        samples_dict={"U-TP":1,"R-TP":1,"M-TP":1,
                  "U-FP":1,"R-FP":1,"M-FP":1,
                  "U-FN":1,"R-FN":1,"M-FN":1}
        
        metrics_dict={"urecall":0.0,"uprecision":0.0,"uf_beta":0.0,
                  "rrecall":0.0,"rprecision":0.0,"rf_beta":0.0,
                  "mrecall":0.0,"mprecision":0.0,"mf_beta":0.0}


        for e1,e2,e3 in zip(inputs,labels,predictions):
          doc = nlp(tokenizer.decode(e1,skip_special_tokens=True))
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          orig = annotator.parse(output)

          ################################################
          doc = nlp(tokenizer.decode(e2,skip_special_tokens=True))
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          cor = annotator.parse(output)

          ################################################
          doc = nlp(tokenizer.decode(e3,skip_special_tokens=True))
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          hyp = annotator.parse(output)

          ################################################
          edit_gold = annotator.annotate(orig, cor)
          edit_hyp = annotator.annotate(orig, hyp)
          app=""
          for e1 in edit_hyp:
            found=False
            for e2 in edit_gold:
              if e1.o_start==e2.o_start and e1.o_end==e2.o_end and e1.o_str==e2.o_str:
                app=str(e2.type)[0]
                found=True
            if not found:
              samples_dict[str(e1.type)[0]+"-FP"]+=1
              FP+=1
            else:
              samples_dict[app+"-TP"]+=1
              found=False
              TP+=1
          for e1 in edit_gold:
            found=False
            for e2 in edit_hyp:
              if e1.o_start==e2.o_start and e1.o_end==e2.o_end and e1.o_str==e2.o_str:
                found=True
            if not found:
              samples_dict[str(e1.type)[0]+"-FN"]+=1
              FN+=1

        ################################################
        # Recall
        recall = TP / (TP + FN)
        # Precision
        precision = TP / (TP + FP)
        # # F0.5
        f_beta = (1 + 0.5 ** 2) * (precision * recall) / ((0.5 ** 2 * precision) + recall)
        print(precision,recall,f_beta)
        self.f05_monitor=f_beta

        # Recall/Prec/F0.5
        
        #Unnecessary
        metrics_dict["urecall"] = samples_dict["U-TP"] / (samples_dict["U-TP"] + samples_dict["U-FN"])
        metrics_dict["uprecision"] = samples_dict["U-TP"] / (samples_dict["U-TP"] + samples_dict["U-FP"])
        metrics_dict["uf_beta"] = (1 + 0.5 ** 2) * (metrics_dict["uprecision"] * metrics_dict["urecall"]) / ((0.5 ** 2 * metrics_dict["uprecision"]) + metrics_dict["urecall"])
        #Replacement
        metrics_dict["rrecall"] = samples_dict["R-TP"] / (samples_dict["R-TP"] + samples_dict["R-FN"])
        metrics_dict["rprecision"] = samples_dict["R-TP"] / (samples_dict["R-TP"] + samples_dict["R-FP"])
        metrics_dict["rf_beta"] = (1 + 0.5 ** 2) * (metrics_dict["rprecision"] * metrics_dict["rrecall"]) / ((0.5 ** 2 * metrics_dict["rprecision"]) + metrics_dict["rrecall"])
        #Missing
        metrics_dict["mrecall"] = samples_dict["M-TP"] / (samples_dict["M-TP"] + samples_dict["M-FN"])
        metrics_dict["mprecision"] = samples_dict["M-TP"] / (samples_dict["M-TP"] + samples_dict["M-FP"])
        metrics_dict["mf_beta"] = (1 + 0.5 ** 2) * (metrics_dict["mprecision"] * metrics_dict["mrecall"]) / ((0.5 ** 2 * metrics_dict["mprecision"]) + metrics_dict["mrecall"])


        result_general={"TP": TP,"FP": FP,"FN": FN,"epoch": self.current_epoch,
                  "Precision":precision,"Recall":recall,"F0.5":f_beta}
        result_general.update(samples_dict)
        result_general.update(metrics_dict)
        wandb.log(result_general)
        result_general={}
        return f_beta

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=self.lr, betas=(0.9, 0.998),eps=1e-08)
        #Schedulers
        #self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=20,start_factor=1.0,end_factor=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer1,lr_lambda=self.l_lambda1)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', factor=0.5, patience=2)
        #print(self.scheduler.state_dict())
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'epoch',  # Adjust the lr at the end of each epoch
                'frequency': 1  # How many epochs to wait before adjusting the lr
                #'monitor' : 'valid_loss'
            }
        }
    
    def on_train_epoch_end(self):
      ##################### Linear #####################
      param=self.scheduler.state_dict()
      param["learning_rate"]=self.scheduler.get_last_lr()[0]
      param["epoch"]=self.current_epoch
      print(param)
      wandb.log(param)
      self.scheduler.step()
      
    def get_predictions_on_end(self,out,pred):
        predictions=[]
        lbl=[]
        inp=[]
        for elem,batchs in zip(out,pred):
          for sent in batchs:
            predictions.append(sent)
          for e in elem[0]:
            lbl.append(e)
          for e in elem[1]:
            inp.append(e)
        return predictions,lbl,inp
    
    def verbose_output(self,inputs_sentences,lbl_sentences,prediction_sentences):
        f = open('predictions'+str(self.current_epoch)+'.txt', 'w', encoding="utf-8")
        for elem in prediction_sentences:
          doc = nlp(tokenizer.decode(elem,skip_special_tokens=True))
          #doc = tokenizer.decode(elem,skip_special_tokens=True)
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          #output=doc
          f.write(output + '\n')
          #f.write(tokenizer.decode(elem,skip_special_tokens=True)+"\n")
        f.close()
        dataset2 = wandb.Artifact('predict', type='dataset')
        dataset2.add_file('predictions'+str(self.current_epoch)+'.txt')
        wandb.log_artifact(dataset2)
        
        f = open('inputs.txt', 'w', encoding="utf-8")
        for elem in inputs_sentences:
          doc = nlp(tokenizer.decode(elem,skip_special_tokens=True))
          #doc = tokenizer.decode(elem,skip_special_tokens=True)
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          #output=doc
          f.write(output + '\n')
          #f.write(tokenizer.decode(elem,skip_special_tokens=True)+"\n")
        f.close()
        dataset1 = wandb.Artifact('original', type='dataset')
        dataset1.add_file('inputs.txt')
        wandb.log_artifact(dataset1)

        f = open('labels.txt', 'w', encoding="utf-8")
        for elem in lbl_sentences:
          doc = nlp(tokenizer.decode(elem,skip_special_tokens=True))
          #doc = tokenizer.decode(elem,skip_special_tokens=True)
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          #output=doc
          f.write(output + '\n')
          #f.write(tokenizer.decode(elem,skip_special_tokens=True)+"\n")
        f.close()
        dataset3 = wandb.Artifact('correct', type='dataset')
        dataset3.add_file('labels.txt')
        wandb.log_artifact(dataset3)

    def on_validation_epoch_end(self):
        print("writing...")
        self.my_predictions =[]
        outpt = self.outputs
        self.outputs=[]
        print(len(outpt),len(self.validation_predictions))
        self.my_predictions,self.my_labels,self.my_inputs=self.get_predictions_on_end(outpt,self.validation_predictions)
        print(len(self.my_predictions),len(self.my_labels),len(self.my_inputs))
        self.validation_predictions=[]
        self.f05_monitor=self.get_F05(self.my_inputs,self.my_labels,self.my_predictions)
        if self.verbose_m2:
          print("writing files...")
          self.verbose_output(self.my_predictions,self.my_labels,self.my_inputs)
        ################################# Table #################################
        wandb_log_data = list(zip([tokenizer.decode(elem,skip_special_tokens=True) for elem in self.my_inputs],
                                    [tokenizer.decode(elem,skip_special_tokens=True) for elem in self.my_predictions],
                                    [tokenizer.decode(elem,skip_special_tokens=True) for elem in self.my_labels]))

        #torch.save(self.model,'/media/models/backup_epoch.pt')
        torch.onnx.export(self.model, torch.randn(1, 100), 'model.onnx')
        wandb.log({"Table1": wandb.Table(data=wandb_log_data, columns=["Original" ,"Prediction", "Correct"]),"F05":self.f05_monitor}, commit=True)
        self.my_inputs=[]
        self.my_labels=[]
        
    def on_test_epoch_end(self):
        print("writing for testing...")
        self.my_predictions =[]
        outpt = self.outputs
        self.outputs=[]
        print(len(outpt),len(self.testing_predictions))
        self.my_predictions,self.my_labels,self.my_inputs=self.get_predictions_on_end(outpt,self.testing_predictions)
        print(len(self.my_predictions),len(self.my_labels),len(self.my_inputs))
        self.testing_predictions=[]
        self.f05_monitor=self.get_F05(self.my_inputs,self.my_labels,self.my_predictions)
        if self.verbose_m2:
          print("writing files...")
          self.verbose_output(self.my_predictions,self.my_labels,self.my_inputs)
        wandb_log_data = list(zip([tokenizer.decode(elem,skip_special_tokens=True) for elem in self.my_inputs],
                                    [tokenizer.decode(elem,skip_special_tokens=True) for elem in self.my_predictions],
                                    [tokenizer.decode(elem,skip_special_tokens=True) for elem in self.my_labels]))
        
        #torch.save(self.model,'/media/models/backup_epoch.pt')
        torch.onnx.export(self.model, torch.randn(1, 100), 'model.onnx')
        wandb.log({"Table1": wandb.Table(data=wandb_log_data, columns=["Original" ,"Prediction", "Correct"]),"F05":self.f05_monitor}, commit=True)
        self.my_inputs=[]
        self.my_labels=[]
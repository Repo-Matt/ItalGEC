import torch
def get_batches_dataset_division(train,corr,trainm,corrm,num_tokens):
  dataset = {"input_ids":[],"attention_mask":[]}
  correct = {"input_ids":[],"attention_mask":[]}
  app_batch1=[]
  app_batch2=[]
  app_batch3=[]
  app_batch4=[]
  counter=0
  max_len=0
  # train
  for s1,s2,s3,s4 in zip(train,trainm,corr,corrm):

    if max_len<max(s1.shape[-1],s3.shape[-1]) and max_len!=0:
      max_len=max(s1.shape[-1],s3.shape[-1])
      for i in range(len(app_batch1)):
        len_previous=max(max_len-app_batch1[i].shape[-1],max_len-app_batch3[i].shape[-1])
        counter+=len_previous

    elif max_len!=0:
        len_previous=max(max_len-s1.shape[-1],max_len-s3.shape[-1])
        counter+=len_previous

    if counter+max(s1.shape[-1],s3.shape[-1])<=num_tokens and counter!=0:
      app_batch1.append(s1.squeeze())
      app_batch2.append(s2.squeeze())
      app_batch3.append(s3.squeeze())
      app_batch4.append(s4.squeeze())

      for i in range(len(app_batch1)):
        #len_previous=max(max_len-app_batch1[i].shape[-1],max_len-app_batch3[i].shape[-1])
        #if len_previous==0:
        #  len_previous+=1
        app_batch1[i]=torch.cat((app_batch1[i],torch.ones(max_len-app_batch1[i].shape[-1])),-1)
        app_batch2[i]=torch.cat((app_batch2[i],torch.ones(max_len-app_batch2[i].shape[-1])),-1)
        app_batch3[i]=torch.cat((app_batch3[i],torch.ones(max_len-app_batch3[i].shape[-1])),-1)
        app_batch4[i]=torch.cat((app_batch4[i],torch.ones(max_len-app_batch4[i].shape[-1])),-1)
      counter+=max(s1.shape[-1],s3.shape[-1])
      
    elif counter==0:
      app_batch1=[s1.squeeze()]
      app_batch2=[s2.squeeze()]
      app_batch3=[s3.squeeze()]
      app_batch4=[s4.squeeze()]
      # just save the longest sentence
      max_len=max(s1.shape[-1],s3.shape[-1])
      # give me also the count of lenghts to check for the 256 complessive
      counter=max_len
      
    elif counter+max(s1.shape[-1],s3.shape[-1])>=num_tokens and counter!=0:
      dataset['input_ids'].append(torch.stack(app_batch1).type(torch.int64))
      dataset['attention_mask'].append(torch.stack(app_batch2).type(torch.int64))
      correct['input_ids'].append(torch.stack(app_batch3).type(torch.int64))
      correct['attention_mask'].append(torch.stack(app_batch4).type(torch.int64))
      max_len=0
      counter=0
  print(dataset['input_ids'][9].shape)
  """
  
  for e1,e2,e3,e4 in zip(
                    dataset['input_ids'],
                    correct['input_ids'],
                    dataset['attention_mask'],
                    correct['attention_mask']):
      for a1,a2,a3,a4 in zip( e1,e2,e3,e4 ):
        if a1.shape[-1]>a2.shape[-1]:
          len_add=a1.shape[-1]-a2.shape[-1]
          a1=torch.cat((a1,torch.ones(len_add)),-1)
          a3=torch.cat((a3,torch.ones(len_add)),-1)
        elif a1.shape[-1]<a2.shape[-1]:
          len_add=a2.shape[-1]-a1.shape[-1]
          a2=torch.cat((a2,torch.ones(len_add)),-1)
          a4=torch.cat((a4,torch.ones(len_add)),-1)
          """
  return dataset,correct
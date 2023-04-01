total=0
for e1,e2,e3,e4 in zip(
                    dataset1['input_ids'],
                    correct1['input_ids'],
                    dataset1['attention_mask'],
                    correct1['attention_mask']):
     
     print(e1.shape,e2.shape,e3.shape,e4.shape)
     for a1,a2,a3,a4 in zip( e1,e2,e3,e4 ):
        #print("\n")
        #print(a1.shape,a2.shape,a3.shape,a4.shape)
        break
        #print("\n ORIGINAL")
        #print(tokenizer.decode(a1.type(torch.int64)))
        #print("\n CORRECT")
        #print(tokenizer.decode(a2.type(torch.int64)))
        total+=a1.shape[0]
     print("total: ",total)
     total=0
     print("\n\n#########\n\n")
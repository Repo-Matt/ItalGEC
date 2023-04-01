import re
f = open('original_gold.m2','r')
train=open('original_train.m2','w')
test=open('original_test.m2','w')
dev=open('original_dev.m2','w')
#correct=open('correct_gold_std.txt','w')
#sentences=[]

orig=[]
numFiles=0
my_sentences=[]
flag=True
for lin in f.readlines():
    if lin[0]=='S':
        document = lin[1:].replace("\n"," ")
        edits = []
        numFiles+=1
    if lin[0]=='A':
        app_strings=lin[:-1].split("|||")[0].replace(" ","|||")+"|||"+"|||".join(lin[:-1].split("|||")[1:])
        app_sent=app_strings.split("|||")
        edits.append(app_sent)
    if lin[0]=='\n':
        my_sentences=[]
        ms=[]
        app_ed=-2
        created_edit=[]
        ce=[]
        in_span=False
        complessive_len=[]
        for i,scan in enumerate(document.split(" ")):
            # checking over the edits for each TOKEN!!!
            whattodo="add"
            for ed in edits:
                if  (i-1 in range(int(ed[1]),int(ed[2])) or i-1==int(ed[1])) and i-1>-1 and (scan=="." or scan=="!" or scan=="?") and ("." not in ed[4] or "?" not in ed[4] or "!" not in ed[4]):
                    whattodo="add"
                    ce.append(ed)
                    break

                elif (i-1 in range(int(ed[1]),int(ed[2])) or i-1==int(ed[1])) and i-1>-1 and (scan!="." or scan!="!" or scan!="?") and ("." in ed[4] or "?" in ed[4] or "!" in ed[4]) and app_ed>0:
                    whattodo="add"
                    ce.append(ed)
                    app_ed=int(ed[2])-int(ed[1])-1
                    break

                elif i-1 not in range(int(ed[1]),int(ed[2])) and (scan=="." or scan=="!" or scan=="?") and ("@" not in document.split(" ")[i+1:i+5] or "@" not in document.split(" ")[i-3:i-1]):
                    whattodo="split"

                elif (i-1 in range(int(ed[1]),int(ed[2])) or i-1==int(ed[1])) and i-1>-1 and (scan!="." or scan!="!" or scan!="?") and ("." not in ed[4] or "?" not in ed[4] or "!" not in ed[4]):
                    whattodo="add"
                    ce.append(ed)
                    
            ms.append(scan)
            if whattodo=="split" or app_ed==0:
                app_ed=-2
                created_edit.append(ce)
                ce=[]
                app_ms=' '.join(ms)
                my_sentences.append(app_ms)
                complessive_len.append(len(ms))
                ms=[]
                
            app_ed-=1
            
        orig+=my_sentences
        app_e=""
        app_len=0
        #print(len(my_sentences),len(created_edit),len(complessive_len))
        for o,edc,lenghts in zip(my_sentences,created_edit,complessive_len):
            if numFiles>81 and numFiles<162:
                result=dev
            elif numFiles<81:
                result=test
            else:
                result=train
            #if o[0]==" ":
            #    result.write("S"+o+"\n")
            #else:
            o="S "+o
            o=o.replace("  "," ")
            result.write(o+"\n")
            #correct.write(o+"\n")
            if not edc:
                result.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
            else:
                for e in edc:
                    if e!=app_e:
                        if int(e[1])!=-1:
                            if app_len!=0:
                                e[1]=";."+str(max(int(e[1])+1-app_len,0))
                                e[2]=";."+str(max(int(e[2])+1-app_len,0))
                            else:
                                e[1]=";."+str(max(int(e[1])-app_len,0))
                                e[2]=";."+str(max(int(e[2])-app_len,0))
                        my_string=str('|||'.join(e)).replace("|||;."," ")+"\n"
                        result.write(my_string)
                    app_e=e
            result.write("\n")
            app_len+=lenghts

#print("len:  ",len(orig),orig[0][0])
#for i,it in enumerate(orig):
#    if it[0]==" ":
#        orig[i]=orig[i][1:]                
#split=open('split_gold.txt','w')
#for elem in orig:
#    split.write(elem+"\n")
    
#print(len(orig),len(edits))
#print(orig[2320],"\n",edits[2320])
#print(orig[2320].split(" ")[5:])
#print(sentences)

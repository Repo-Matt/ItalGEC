

import os
import re
import spacy

# Set directory path where files are stored
directory_path = "./"

# Load the Spacy English model
nlp = spacy.load("it_core_news_sm")
original = open("correct_gold.txt","r")
reference = open("correct_gold.txt","r")



########################################################

original_list=[]
reference_list=[]
num=0
for read in reference.readlines():
    reference_list.append(read[:-1])

for read in original.readlines():
    if read[-1]==" " or read[-1]=="\n":
        if len(read)>3:
            if read[-2]==" " or read[-2]=="\n":
                original_list.append(read[:-2])
            else:
                original_list.append(read[:-1])

for s in original_list:
    if re.search(r'\.[^\s]',s)!=None:
        print(s)

"""
for scan in reference_list:
    if scan not in original_list and "unreadable" not in scan and not ". ." in scan and not "! !" in scan:
        if "! ?" not in scan:
            if "? !" not in scan:
                #print(scan)
        num+=1
        
"""
print(num)

########################################################

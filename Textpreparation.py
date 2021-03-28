'''
Created on 26 мар. 2021 г.

@author: romansmirnov
'''
import os
import re
import numpy as np
#STRING TO SENTENCE
import TextExtract

def get_legal_parts(folder, separator = '_/separator/_', save_path = 'results.txt'):
    n_e=0
    n_s=0
    baskreg_files = os.listdir(folder)
    fulltext = []
    for file in baskreg_files:
        with open(folder+file, 'r') as rf:
            fulltext.append(rf.read())
    texts = str(separator).join(fulltext).split(separator)
    beginnings = []
    endings = []
    for text in texts:
        end = re.search('п р и г о в о р и л|приговорил|постановил|п о с т а н о в и л', text.lower())
        start = re.search('установил|у с т а н о в и л', text.lower())
        if end is not None:
            endings.append(TextExtract._from_string_to_string(re.sub(' +', ' ', re.sub('\n', ' ',text[end.start():])).strip()))
        else:
            n_e +=1
            endings.append(' ')
            
        if start is not None:
            
            beginnings.append(TextExtract._from_string_to_string(re.sub(' +', ' ', re.sub('\n', ' ', text[:start.start()])).strip()))
        else:
            n_s +=1
            endings.append(' ')
    result = np.unique([b + '\n' + e for b,e in zip(beginnings,endings)]).tolist()
    print(len(result))
    print(n_s)
    print(n_e)
    with open(folder+save_path, 'w') as sf:
        sf.write(('\n').join(result))

get_legal_parts('./koap_extra/', '_/separator/_')





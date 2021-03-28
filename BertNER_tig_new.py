'''
Created on 27 март. 2021 г.

@author: romansmirnov



For any NER task, we need a sequence of words and their corresponding labels. To extract features for these words from BERT, they need to be tokenized into subwords.

For example, the word 'infrequent' (with label B-count) will be tokenized into ['in', '##fr', '##e', '##quent']. How will its label be represented?

According to the BERT paper, "We use the representation of the first sub-token as the input to the token-level classifier over the NER label set".

So I assume, for the subwords ['in', '##fr', '##e', '##quent'] , the label for the first subword will either be this ['B-count', 'B-count', 'B-count', 'B-count'] where we propagate the word label to all the subwords. Or should it be ['B-count', 'X', 'X', 'X'] where we leave the original label on the first token of the word, then use the label “X” for subwords of that word.

Any help will be appreciated.

#ANSWER: Method 2 is the correct one.

Leave the actual label of the word only in the first sub-token, and the other sub-tokens will have a dummy label (which in this case is 'X'). The important thing is that when calculating the loss (e.g., CELoss) and metrics (e.g., F1), this 'X' labels on the sub-tokens are not taken into account.

This is also the reason why we don't use method 1 is that otherwise, we would be introducing more labels of the type [B-count] and affecting the support number for such a class (which would make a test set no longer comparable with other models that do not increase the number of labels for such class).

#ANSWER: We have been trying to do the same thing. One thing we tried is tagging the sequences after wordpiece tokenization.
So in our case

Jim    Hen    ##son  was  a puppet  ##eer  
 B-PER  I-PER  X     O    O   O       X
becomes

Jim    Hen    ##son  was  a puppet  ##eer    
B-PER  I-PER  I-PER  O    O   O        O
And while decoding we merge the tags for subtokens of a token like this.

def convert_to_original_length(sentence, tags):
   r = []
   r_tags = []
   for index, token in enumerate(tokenizer.tokenize(sentence)):
       if token.startswith("##"):
           if r:
               r[-1] = f"{r[-1]}{token[2:]}"
       else:
           r.append(token)
           r_tags.append(tags[index])
   return r_tags
We found it work better than taking the tag of first subtoken.


###My SOLUTION###

-100 label for cls sep and pad tokens OR it should be other class "ignore index of -100" - http://sujitpal.blogspot.com/2020/01/adding-transformer-based-ner-model-into.html
Subtokens with I-MARK
INCLUDE SUBTOKENS IN METRICS
'''
import TextExtract

import re
import json
import os

from transformers import BertForTokenClassification, AdamW
from transformers import BertTokenizer

import torch

import numpy as np
from random import shuffle


def get_train_data(path, tokenizer, mode = None):
    with open(path, 'r') as file:
        data = file.read()
    blocks = data.split('\n\n')
    x_data = []
    y_data = []
    
    for_labels = []
    
    for block in blocks:
        x = []
        y = []
        sb = block.split('\n')
        for s in sb:
            if s!='':
                t = s.split(' ')
                x.append(t[0])
                
                y.append(t[1])
                for_labels.append(t[1])
                
                if len(tokenizer(t[0])["input_ids"])>3:
                    for _ in range(len(tokenizer(t[0])["input_ids"])-3):
                        if t[1]!='O':
                            if mode is None:
                                y.append('I'+t[1][1:])
                                for_labels.append('I'+t[1][1:])
                            else:
                                if mode==-100:
                                    y.append(mode)
                                    #for_labels.append(mode)
                                else:
                                    y.append(mode)
                                    for_labels.append(mode)
                        else:
                            y.append(t[1])
                            for_labels.append(t[1])
        if x!=[] and y!=[]:
            x_data.append(x)
            y_data.append(y)
                
                
                
    labels = np.unique(for_labels).tolist()
    print(labels)
    i_y_data = []
    for y in y_data:
        i_y = [labels.index(x) if x!=-100 else -100 for x in y]
        i_y_data.append(i_y)
    return x_data, i_y_data, {k: v for k,v in enumerate(labels)}


def get_data(path, tokenizer, mode = None):
    with open(path, 'r') as file:
        data = file.read()
    blocks = data.split('\n\n')
    x_data = []
    y_data = []
    
    
    for block in blocks:
        x = []
        y = []
        sb = block.split('\n')
        for s in sb:
            if s!='':
                t = s.split(' ')
                x.append(t[0])
                
                y.append(t[1])
                
                
                if len(tokenizer(t[0])["input_ids"])>3:
                    for _ in range(len(tokenizer(t[0])["input_ids"])-3):
                        if t[1]!='O':
                            if mode is None:
                                y.append('I'+t[1][1:])
                            else:
                                y.append(mode)
                            
                        else:
                            y.append(t[1])
                            
        if x!=[] and y!=[]:
            x_data.append(x)
            y_data.append(y)
    
    return x_data, y_data

def create_batches(x,pre_y,a_m,batch_size=2,do_shuffle=True, special_classes={}):
    if len(special_classes) != 0:
        sp_c = int(list(special_classes.keys())[0])
    else:
        sp_c = -100
    x = x.tolist()
    a = a_m.tolist()
    y = []
    l = len(x[0])
    for obj in pre_y:
        t = [sp_c]#were zeros
        t.extend(obj)
        if len(t)<l:
            t_a = np.repeat(sp_c, l-len(t))
            if len(t_a)>1:
                t.extend(t_a)
            else:
                t.append(sp_c)
        else:
            t = t[:512]
            print(len(t))
            print('somehow more - error?')
        y.append(t)
    
    if do_shuffle:
        t_s = [[i,j,k] for i,j,k in zip(x,y,a)]
        #print(t_s)
        shuffle(t_s)
        x = [i[0] for i in t_s]
        y = [i[1] for i in t_s]
        a = [i[2] for i in t_s]
        
    x_batches = []
    y_batches = []
    a_batches = []
    
    i=0
    while i<len(x):
        if i <len(x)-batch_size:
            x_batches.append(x[i:i+batch_size])
            y_batches.append(y[i:i+batch_size])
            a_batches.append(a[i:i+batch_size])
            i+=batch_size
        else:
            x_batches.append(x[i:])
            y_batches.append(y[i:])
            a_batches.append(a[i:])
            i+=batch_size
    return x_batches, y_batches, a_batches


def train_BERT_NER(save_dir, load_mod = 'bert-base-uncased', tokenizer_path = '/content/drive/MyDrive/rubert_cased_L-12_H-768_A-12_pt', train_txt_path = './NER_train.txt', labels_path = './labels_dictionary_big.json', n_used = None, cont = None):
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    train_dataset_x, train_dataset_y, labels_dictionary = get_train_data(train_txt_path, tokenizer, mode = cont)
    
    if n_used is not None:
        special_classes = {f'{len(labels_dictionary)-1}': str(n_used)}
        labels_dictionary[str(len(labels_dictionary))] = str(n_used)
    else:
        special_classes = {}
    
    with open(labels_path, 'w') as outfile:
        json.dump(labels_dictionary, outfile)
        
    
    #device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = BertForTokenClassification.from_pretrained(load_mod, num_labels=len(labels_dictionary)) #should it be +1 because of -100
    model.to(device)
    model.train()
    
    
    
    tokenizings = tokenizer(train_dataset_x, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True, max_length=512)
    
    x = tokenizings["input_ids"]
    
    a_m = tokenizings['attention_mask']
    
    x_t, y_t, a_t = create_batches(x, train_dataset_y, a_m, batch_size=2, do_shuffle=False,special_classes=special_classes)
    
    
    optim = AdamW(model.parameters(), lr=5e-5)
    
    '''
    #Вопрос: нужны ли классы для [CLS] и [SEP]? Может их ставить в -100?
    '''
    n_steps = len(x_t)

    for epoch in range(8):
        n_step = 0
        print(epoch)
        for batch, out, attention in zip(x_t, y_t, a_t):
            optim.zero_grad()
            input_ids = torch.tensor(batch)
            attention_mask = torch.tensor(attention)
            labels = torch.tensor(out)
            outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
            loss = outputs[0]
            loss.backward()
            optim.step()
            if n_step % 100 == 0:
                print(loss)
            n_step+=1
            print(f'{n_step} out of {n_steps}')

        #if epoch % 3 == 1 or epoch > 10:
        #    save_dir_cur = save_dir + str(epoch)#os.path.join(save_pth_path, 'Bert_NER_mod' + str(epoch)) 
        #    model.save_pretrained(save_dir_cur)
            
    
    model.eval()
    model.save_pretrained(save_dir)
    
    
def calc_score(a_lab, b_lab):
    #print(len(a_lab), len(b_lab))
    alab = [i if len(i)==1 else i[2:] for i in a_lab]
    blab = [i if len(i)==1 else i[2:] for i in b_lab]
    
    #print(blab)

    if len(alab)>len(blab):
        alab = alab[0:len(blab)]
        
    else:
        blab = blab[0:len(alab)]
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for x,y in zip(alab,blab):
        if x!='O' and y!='O':
            TP+=1
        if x=='O' and y!='O':
            FP+=1
        if x =='O' and y=='O':
            TN+=1
        if x!='O' and y=='O':
            FN+=1
    return TP,FP,TN,FN
    
def use_BERT_NER(load_dir, tokenizer_path = 'bert-base-uncased', labels_dictionary_path='labels_dictionary', mode='eval', test_txt_path = './NER_test.txt', req = None, cont = None):
    
    #device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained(load_dir)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model.to(device)
    model.eval()
    
    with open(labels_dictionary_path) as json_file:
        labels_dictionary = json.load(json_file)
    
    if mode=='predict':
        if req!=None:
            print(TextExtract._from_string_to_string(req))
            x = tokenizer(TextExtract._from_string_to_string(req).split(), return_tensors='pt', padding=True, truncation=True, is_split_into_words=True, max_length=512).to(device)
            #print(x)
            outputs = model(**x)
            #print(outputs)
            answer = outputs.logits.tolist()
            #print(answer)
            
            r_i = []
            for t in x['input_ids'][0]:
                r_i.append(tokenizer.decode(t))
            print(r_i)

            result_labels = []
            for i,a in enumerate(answer[0]):
                am = max(a)
                im = a.index(am)
                result_labels.append(labels_dictionary[str(im)])
            if len(r_i) == len(result_labels):
                print([[k,v] for k,v in zip(r_i, result_labels)])
    if mode == 'eval':
    
        test_dataset_x, test_dataset_y = get_data(test_txt_path, tokenizer, mode = cont)
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        #print(labels_dictionary)
        for i,x in enumerate(test_dataset_x):        
            x = tokenizer(x, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True, max_length=512)
            x = x.to(device)
            outputs = model(**x)
            answer = outputs.logits.tolist()
            
            result_labels = []
            for a in answer[0]:
                am = max(a)
                im = a.index(am)
                result_labels.append(labels_dictionary[str(im)])
            
            to_calc_score = ['O']
            to_calc_score.extend(test_dataset_y[i])
            to_calc_score.extend(['O'])
            #['O'].extend(test_dataset_y[i]).extend(['O'])
            #to_calc_score.extend(['O'])
            tTP,tFP,tTN,tFN = calc_score(to_calc_score, result_labels)
            TP+=tTP
            FP+=tFP
            TN+=tTN
            FN+=tFN
            
        print(f'Precision = {TP/(TP+FP)} Recall = {TP/(TP+FN)} F1 = {2*(TP/(TP+FP)*TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))}')
            

train_BERT_NER('/content/drive/MyDrive/Bert/Bert_NER_mod', load_mod = '/content/drive/MyDrive/Bert/RuBert',
               tokenizer_path = '/content/drive/MyDrive/Bert/RuBert', train_txt_path = './NerTrain.txt', 
               labels_path = './labels_dictionary_big.json', n_used = None, cont = None)
#n_used - label for not used tokens in X
#cont - label for continious tags in Y

use_BERT_NER('./Bert_NER_mod', './RuBert', 
             './labels_dictionary_big.json', mode='eval',  test_txt_path = './NerTrain.txt', cont = None)


req = "П О С Т А Н О В Л Е Н �? Е по делу об административном правонарушении 08 сентября 2010 г.с.Сибас РБ Мировой судья судебного участка по Шаранскому району Республики Мордовия Дубовцев А.А., расположенный по адресу: Ленинградская область, с.Шаран, ул.Свердлова, д.1, каб.17, рассмотрев в открытом судебном заседании дело об административном правонарушении, предусмотренном ч.2 ст.14.1 КоАП РФ в отношении �?сламбратова <Ф�?О1>, <ДАТА2> г.р., урож. <АДРЕС> зарег. и прож. по адресу: <АДРЕС>, <ОБЕЗЛ�?ЧЕНО>"
#req = "ПРИГОВОРИЛ : Гарееву З . К . признать виновной в совершении преступления , предусмотренного ч . 1 ст . 171 УК РФ , назначив ей наказание в виде штрафа в размере 50 000 (пятидесяти тысяч ) рублей . Меру пресечения Гареевой З . К . – подписку о невыезде и надлежащем поведении – оставить без изменения до вступления приговора в законную силу . Вещественные доказательства по делу : 266 бутылок алкогольной продукции – уничтожить"
use_BERT_NER('./Bert_NER_mod', './RuBert', 
             './labels_dictionary_big.json', mode='predict', req = req, test_txt_path = './NER_test.txt', cont = None)










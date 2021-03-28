'''
Created on 28 мар. 2021 г.

@author: romansmirnov
'''

from flask import Flask, request, jsonify

import json
import re

import torch

from transformers import BertForTokenClassification
from transformers import BertTokenizer

BERT_DIR = './Bert_NER_1'
TOKENIZER_DIR = './Bert_NER_1'
LABELS_DICT_DIR = './labels_dictionary_big.json'

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    answer = {'data': 'С Иванова И (ОТВЕТЧИК_ПОВТОР) . (ОТВЕТЧИК_ПОВТОР) И (ОТВЕТЧИК_ПОВТОР) . взыскать штраф в пользу бюджета в размере 10 (ПРИГОВОР) 000 (ПРИГОВОР) (десяти (ПРИГОВОР) тысяч (ПРИГОВОР) ) (ПРИГОВОР) рублей (ПРИГОВОР)'}

    #answer['data'] = predict_BERT_NER('С Иванова И.И. взыскать штраф в пользу бюджета в размере 10 000 (десяти тысяч) рублей ')
    #result = answer = {'data': 'С Иванова И (ОТВЕТЧИК_ПОВТОР) . (ОТВЕТЧИК_ПОВТОР) И (ОТВЕТЧИК_ПОВТОР) . взыскать штраф в пользу бюджета в размере 10 (ПРИГОВОР) 000 (ПРИГОВОР) (десяти (ПРИГОВОР) тысяч (ПРИГОВОР) ) (ПРИГОВОР) рублей (ПРИГОВОР)'}

    print(answer)
    pass

@app.route('/gen', methods=['POST'])
def gen():
    text = request.json['data']
    answer = {}
    answer['data'] = predict_BERT_NER(text)
    return jsonify(answer)


def _from_string_to_string(string):
    def _addspace(matchobj):
        def _define_class(obj, cl):
            for k,v in cl.items():
                if re.sub(v, '', obj) == '':
                    return k
        classes = {'ra': '[А-Яа-я]', 'ea': '[A-Za-z]', 'pu': '[^ \w]', 'di': '\d', 'sp': ' '}
        data_classes = []
        for a in matchobj.group(0):
            data_classes.append(_define_class(a,classes))
        res = ''
        for i,obj in enumerate(data_classes):
            res+=matchobj.group(0)[i]
            if i<len(data_classes)-1:
                if obj!=data_classes[i+1]:
                    res+=' '
        return res
    
    string = ' ' + string + ' '
    p_string = re.sub(' ', '  ', string)
    while re.search('.[^ \w].|\d\D.|.\D\d|[A-Za-z][А-Яа-я]|[А-Яа-я][A-Za-z]', string)!=None:
        string = re.sub(' ', '  ', string)
        string = re.sub('.[^ \w].|\d\D.|.\D\d|[A-Za-z][А-Яа-я]|[А-Яа-я][A-Za-z]',_addspace,string)
        string = re.sub(' +', ' ', string)
        if p_string == string:
            break
        p_string = string
    return re.sub(' +', ' ', string.strip())

def predict_BERT_NER(needed_str, load_dir = BERT_DIR, tokenizer_path = TOKENIZER_DIR, labels_dictionary_path = LABELS_DICT_DIR):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained(load_dir)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model.to(device)
    model.eval()
    
    with open(labels_dictionary_path) as json_file:
        labels_dictionary = json.load(json_file)

    splitted_str = _from_string_to_string(needed_str).split()
    x = tokenizer(splitted_str, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True, max_length=512)
    x = x.to(device)  
    outputs = model(**x)
    answer = outputs.logits.tolist()
    r_i = []
    for t in x['input_ids'][0]:
        r_i.append(tokenizer.decode(t))
    
    result_labels, ans_labels = [], []

    for a in answer[0]:
        am = max(a)
        im = a.index(am)
        result_labels.append(labels_dictionary[str(im)])
        
    try:
        if len(r_i) == len(result_labels):
            for k in range(len(r_i)):
                start_let = r_i[k][0]
                if start_let != '[' and start_let != '#':
                    ans_labels.append(result_labels[k])
            ans = [[k, v] for k,v in zip(splitted_str, ans_labels)]

            ans_str = ''
            for next_word in ans:
                ans_str += (next_word[0] + ' ')
                if next_word[1] != 'O':
                    word_class = next_word[1][2:]
                    ans_str += '(' + word_class + ') '
            ans_str = ans_str[:-1]
            
    except Exception:
        return 'Tokenizing problem'
        pass

    #return ans # list of lists
    return ans_str #string with BIG classes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
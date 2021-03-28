'''
Created on 27 март. 2021 г.

@author: romansmirnov
'''

import re
from nltk.stem import SnowballStemmer
import numpy as np
import json
import math
#import fasttext
from multiprocessing import Process, Manager

def _lower_stem(string, rule, ps):
    if rule['stemming'] and rule['lowercase']:
        string= ' '.join([ps.stem(x).lower() for x in string.split(' ')])
    elif rule['stemming']:
        string= ' '.join([ps.stem(x) for x in string.split(' ')])
    elif rule['lowercase']:
        string= ' '.join([x.lower() for x in string.split(' ')])
    elif rule['stemming']==False and rule['lowercase']==False:
        string= ' '.join([x for x in string.split(' ')])
    return string

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
    
def _get_start_end_rule(match_objs, sp_sentence_words):
    jzlist = []
    for pre_match_obj in match_objs:
        match_obj = pre_match_obj.group(0)
        if match_obj!='':
            match_obj = _from_string_to_string(match_obj)
            n_g = len(match_obj.split(' '))
            j = 0
            jm = len(sp_sentence_words)
            s_n_gramms = []
            while j<jm-n_g+1:
                s_n_gramms.append(' '.join(sp_sentence_words[j:j+n_g]))
                j+=1
            add_flag = False

            for j, s_n_gramm in enumerate(s_n_gramms):
                if match_obj == s_n_gramm:
                    add_flag = True
                    break

            if add_flag:
                z = j+len(s_n_gramm.split(' '))-1
                jzlist.append([j, z])
    return jzlist

def _jzcorrect(base_sentence, rule, _A,ps,jzlist, sp_sentence_words):
    if rule['special_del']!=None and rule['after_del_sentence_transform']==False:
        _B = {'stemming': True, 'lowercase': True}
        _B1 = {'stemming': False, 'lowercase': True}
        sl_base = _lower_stem(_from_string_to_string(base_sentence), rule, ps).split(' ')
        
        original_sentence_parts = []
        sp_i = 0
        r_mode = 0
        part = []
        for word in sl_base:
            if sp_i<len(sp_sentence_words):
                if word == sp_sentence_words[sp_i]:
                    sp_i += 1
                    if r_mode == 0:
                        part.append(word)
                    if r_mode == 1:
                        original_sentence_parts.append(' '.join(part))
                        part = []
                        r_mode = 0
                        part.append(word)
                else:
                    if r_mode == 1:
                        part.append(word)
                    if r_mode == 0:
                        original_sentence_parts.append(' '.join(part))
                        part = []
                        r_mode = 1
                        part.append(word)
        if part!=[]:
            original_sentence_parts.append(' '.join(part))
                    
        z_i = 0
        d_i = 1
        num_borders = []
        for i,part in enumerate(original_sentence_parts):
            if i % 2 == 0:
                objs = len(re.sub(' +', ' ',part).split(' '))-1
                num_borders.append([z_i, z_i+objs, d_i])
                z_i += objs+1
                d_i+=2
        
        rlist = []
        for obj in jzlist:
            _s = obj[0]
            _f = obj[1]
            for bord in num_borders:
                if obj[0]>=bord[0] and obj[1]<=bord[1]:
                    break
                elif obj[0]<=bord[1]:
                    if obj[1]>bord[1]:

                        _f+=len(re.sub(' +', ' ',original_sentence_parts[bord[2]]).split(' '))
                        
                    elif obj[0]<bord[0] and obj[1]<bord[0]:
                        break

                elif obj[0]>bord[0]:
                    _s += len(re.sub(' +', ' ',original_sentence_parts[bord[2]]).split(' '))
                    _f += len(re.sub(' +', ' ',original_sentence_parts[bord[2]]).split(' '))
            rlist.append([_s,_f])
        return rlist
    else:
        return jzlist

def _check_exclude_and_cutting(string, rule):
    if rule['exclude_cond']!=None:
        if re.search(rule['exclude_cond'],string)!=None:
            return string,False
        else:
            return string, True
    if rule['cut_of_result']!=None:
        string = re.sub(' +', ' ', re.sub(rule['cut_of_result'], '', string).strip())
        return string, True
    return string, True

def _extract(sentence, rules, ps):
    _A = {'stemming':False, 'lowercase': False}
    base_sentence = sentence
    
    a_list = []
    
    for rule in rules:
        original_sentence = _from_string_to_string(base_sentence)
        special_sentence = original_sentence
        sentence = ' ' + base_sentence + ' '
        if rule['special_del']!=None:
            to_del = ' '+' | '.join(rule['special_del'])+' '
            #corrected to do search on non normalized
            #sentence = _from_string_to_string(re.sub(to_del, ' ', sentence).strip())
            sentence = re.sub(to_del, ' ', sentence).strip()
            #special_sentence = re.sub(' +', ' ',sentence)
            special_sentence = re.sub(' +', ' ',_from_string_to_string(sentence))
        if rule['stemming']:
            sentence = ' '.join([ps.stem(x) for x in re.split(' ', sentence)]).strip()
        if rule['lowercase']:
            sentence = sentence.lower().strip()
            if isinstance(rule['rule'], list):
                for ri,rp in enumerate(rule['rule']):
                    rule['rule'][ri] = rp.lower()
            else:
                rule['rule'] = rule['rule'].lower()
            
        sentence = re.sub(' +', ' ',sentence)
        
        if rule['special_del']!=None and rule['after_del_sentence_transform']==False:
            original_sentence_words = original_sentence.strip().split(' ')
        else:
            original_sentence_words = special_sentence.strip().split(' ')
        sp_sentence_words = _lower_stem(special_sentence,rule,ps).strip().split(' ')
        
        if rule['method'] == 'regex':
            match_objs = re.finditer(rule['rule'], sentence)
            jzlist = _jzcorrect(base_sentence, rule,_A,ps,_get_start_end_rule(match_objs, sp_sentence_words),sp_sentence_words)
            for obj in jzlist:
                answer, s_status = _check_exclude_and_cutting(' '.join(original_sentence_words[obj[0]:obj[1]+1]),rule)
                if s_status:
                    a_list.append(answer)
                
        if rule['method'] == 'between':
            match_objs = re.finditer(rule['rule'][0], sentence)
            jzlist1 = _jzcorrect(base_sentence, rule,_A,ps,_get_start_end_rule(match_objs, sp_sentence_words),sp_sentence_words)
            match_objs = re.finditer(rule['rule'][1], sentence)
            jzlist2 = _jzcorrect(base_sentence, rule,_A,ps,_get_start_end_rule(match_objs, sp_sentence_words),sp_sentence_words)
            for obj1 in jzlist1:
                for obj2 in jzlist2:
                    if obj2[0]>obj1[1]:
                        if rule['method_data'][0]:
                            answer, s_status =_check_exclude_and_cutting(' '.join(original_sentence_words[obj1[1]-obj1[1]+obj1[0]:obj2[0]+obj2[1]-obj2[0]]),rule)
                            if s_status:
                                a_list.append(answer)
                        else:
                            answer, s_status =_check_exclude_and_cutting(' '.join(original_sentence_words[obj1[1]+1:obj2[0]-1]),rule)
                            if s_status:
                                a_list.append(answer)
                        break
                    
        if rule['method'] == 'after':
            match_objs = re.finditer(rule['rule'], sentence)
            jzlist = _jzcorrect(base_sentence, rule,_A,ps,_get_start_end_rule(match_objs, sp_sentence_words),sp_sentence_words)
            for obj in jzlist:
                if rule['method_data'][1]:
                    answer, s_status =_check_exclude_and_cutting(' '.join(original_sentence_words[obj[0]:obj[1]+1+int(rule['method_data'][0])]),rule)
                    if s_status:
                        a_list.append(answer)
                else:
                    answer, s_status =_check_exclude_and_cutting(' '.join(original_sentence_words[obj[1]+1:obj[1]+1+int(rule['method_data'][0])]),rule)
                    if s_status:
                        a_list.append(answer)
        
        if rule['method'] == 'before':
            match_objs = re.finditer(rule['rule'], sentence)
            jzlist = _jzcorrect(base_sentence, rule,_A,ps,_get_start_end_rule(match_objs, sp_sentence_words),sp_sentence_words)
            for obj in jzlist:
                if rule['method_data'][1]:
                    answer, s_status =_check_exclude_and_cutting(' '.join(original_sentence_words[obj[0]-int(rule['method_data'][0]):obj[0]+obj[1]-obj[0]]),rule)
                    if s_status:
                        a_list.append(answer)
                else:
                    answer, s_status =_check_exclude_and_cutting(' '.join(original_sentence_words[obj[0]-int(rule['method_data'][0]):obj[0]-1]),rule)
                    if s_status:
                        a_list.append(answer)
    return a_list

def extract_on_rules(sentence, rules,ps = SnowballStemmer("russian")):
    #WORKS WITH FULL MATCH OF WORD (NO PARTS)
    #DELETION WORKS ON FULL WORDS
    #IN AFTER AND BETWEEN METHODS PUNCTUATION IS NOT COUNTED
    #REGEX WORKS WITH STRINGS BEFORE SPLITTING, SEPARATING PUNCTUATION
    answers = {} 
    for rule_name, rule in rules.items():
        answers[rule_name] = _extract(sentence, rule, ps)
    
    return answers

def extract_on_dict(input_sentence, dictionary, exclude = ['\([^\)]+\)','\d','[^а-яА-Яa-zA-Z ]'], substring = None):        
    #one key - many potential values

    #1 - value to regex - OPERATE SHORTESTS ('Р.Ш. Мухамедзанов')
    #2 - stemming, lowercase and punctuation (exclude dot) - separation?
    #3 - match on full text or part (if long) beggining & end - or distance of sentences with same start
    
    #SYNONIMIZED CONSTRUCTIONS - NOT REALIZED

    sp_constr = {'spc' : '[а-я]*'}
    
    search_dict = {}
    raw_search_dict = {}
    rule = {'lowercase': True, 'stemming': True, 'special_del' : exclude, 'after_del_sentence_transform':False}
    _A = {'lowercase': False, 'stemming': False}
    ps = SnowballStemmer("russian")
    
    to_del = ' ' + ' | '.join(exclude) + ' '
    
    base_sentence = _from_string_to_string(input_sentence)
    original_sentense_words = base_sentence.split()
    
    for key, value in dictionary.items():
        values = []
        for val in value:
            
            string_tokens = _lower_stem(_from_string_to_string(val),rule,ps).split(' ')
            for i,token in enumerate(string_tokens):
                if i<len(string_tokens)-1:
                    if len(token)==1 and string_tokens[i+1] == '.':
                        string_tokens[i] = string_tokens[i] + 'spc'
            sentence = ' '.join(string_tokens).strip()
            
            if substring!=None:
                sentence = [re.sub(' +',' ', re.sub(to_del, ' ', ' ' + x + ' ').strip()) for x in sentence.split(substring)]
            if isinstance(sentence,list):
                values.extend(sentence)
            else:
                values.append(re.sub(' +',' ', re.sub(to_del, ' ', ' ' +' '.join(string_tokens)+ ' ').strip()))
        search_dict[key] = '|'.join(values)
        raw_search_dict[key] = values
        
    for key, value in sp_constr.items():
        for dkey in search_dict.keys():
            search_dict[dkey] = re.sub(key,value,search_dict[dkey])
    input_sentence=re.sub(' +',' ', re.sub(to_del, ' ',' ' + _lower_stem(_from_string_to_string(input_sentence),rule,ps)+ ' ')).strip(' ')

    sp_sentence_words = input_sentence.split()
    answer_dict = {}
    for key, value in search_dict.items():
        if value!='()' and value!='':
            match_objs = re.finditer(value, input_sentence)
            answer_dict[key] =[]
            jzlist = _jzcorrect(base_sentence, rule,_A,ps,_get_start_end_rule(match_objs, sp_sentence_words),sp_sentence_words)
            for obj in jzlist:
                answer_dict[key].append(' '.join(original_sentense_words[obj[0]:obj[1]+1]))
    #аналог search engine
    #делим на n-gramm по длине целевого, находим минимальное, если одинаковое расстояние у соседей - объединяем
    #EXTRA WORD APPEARING
    for key, values in raw_search_dict.items():
        for value in values:
            w_val = value.split(' ')
            if len(w_val)>=10:
                w_sentence = input_sentence.split(' ')
                w_sentences = [w_sentence[i:i+len(w_val)] for i in range(len(w_sentence)-len(w_val))]
                
                
                l_distances = [_L_distance(w_val,i) for i in w_sentences]
                if min(l_distances)<=3:
                    i_s = []
                    for i in range(len(l_distances)):
                        if l_distances[i]==min(l_distances):
                            i_s.append(i)
                    j0 = i_s[0]
                    starts = [i_s[0]]
                    ends = [i_s[0]+len(w_val)]
                    
                    for j in i_s:
                        if j==j0+1:
                            ends[len(ends)-1] = j+len(w_val)
                        else:
                            starts.append(j)
                            ends.append(j+len(w_val))
                            j0 = j
                    
                    jzlist0 = []
                    for x,y in zip(starts,ends):
                        jzlist0.append([x,y-1])
                        
                    jzlist = _jzcorrect(base_sentence, rule,_A,ps,jzlist0,sp_sentence_words)
                    for obj in jzlist:
                        answer_dict[key].append(' '.join(original_sentense_words[obj[0]:obj[1]+1]))
    return answer_dict
        
def _locate_substring(aim, string):
    l_a = len(aim.split(' '))
    string_words = string.split(' ')
    w_sentences = [' '.join(string_words[i:i+l_a]) for i in range(len(string_words)-l_a)]
    answers = [[i,i+l_a-1] for i,x in enumerate(w_sentences) if x == aim]
    return answers

def _sentence_to_sentences(sentence):
    def _add_separation_token(matchobj):
        def _dot_and_sep(matchobj_dot):
            #print(matchobj_dot.group(0)+ '_separator_')
            return matchobj_dot.group(0) + '_separator_'
        return re.sub('[\.!\?;]',_dot_and_sep, matchobj.group(0))
        
    sentences = []
    sentence = re.sub('[а-яa-z]{3,} ?[\.!\?;]|\d ?\[\.!\?;] ?[А-ЯA-Z]', _add_separation_token, sentence)
    pre_sentences = sentence.split('_separator_')
    
    #print(sentence)
    #print(pre_sentences)
    
    for i,sentence in enumerate(pre_sentences):
        if i+1<len(pre_sentences):
            if re.sub('[^А-Яа-яA-Za-z]','',pre_sentences[i+1])=='':
                sentences.append(sentence+pre_sentences[i+1])
            else:
                sentences.append(sentence)
        else:
            if re.sub('[^А-Яа-яA-Za-z]','',pre_sentences[i])!='':
                sentences.append(sentence)

    '''for i,sentence in enumerate(pre_sentences):
        if i%2 == 0:
            if i+1<len(pre_sentences):
                sentences.append(sentence+pre_sentences[i+1])
            else:
                sentences.append(sentence)'''
    return sentences

def sentence_to_ner(sentence, dictionary, multiple_sent = False):
    if multiple_sent == False:
        sentence = _from_string_to_string(sentence)
        matches = {}
        for key, value in dictionary.items():
            if isinstance(value, list):
                matches[key] = [_from_string_to_string(x) for x in value if _from_string_to_string(x) in sentence]
            else:
                print('Value of the dictionary should be LIST')
                
        #print(matches)
        
        s_w = sentence.split(' ')
        marks = [['O'] for x in range(len(s_w))]
        #marks = np.repeat([0],len(s_w)).tolist()
        for key,value in matches.items():
            if isinstance(value, list) and value!=[]:
                for val in value:
                    positions = _locate_substring(val,sentence)
                    for obj in positions:
                        i1 = obj[0]
                        i2 = obj[1]
                        pre_mark = 'B-'
                        while i1<=i2:
                            #only last one mark
                            #if marks[i1] == ['0']:
                            marks[i1] = [pre_mark+str(key)]
                            #else:
                            #    marks[i1].append(str(key))
                            i1+=1
                            pre_mark = 'I-'
        answer = ''
        for i,word in enumerate(s_w):
            answer += word + ' ' + ' '.join(marks[i])+'\n'
    else:
        #works good if separation - word which length more than 3 symbols and !?.; after
        
        answer = []
        sentences = _sentence_to_sentences(sentence)
        for s in sentences:
            answer.append(sentence_to_ner(s,dictionary,multiple_sent = False))
    return answer

def _remove_extra(string):
    #return re.sub(' +', ' ',re.sub('[^А-Яа-я0-9]',' ',string)).strip()
    return re.sub('[^А-Яа-я0-9]','',string)

def _mark_collection(collection, ps):
    rule = {'stemming': False, 'lowercase': True}
    rule_st = {'stemming': True, 'lowercase': False}
    marked_collection = []
    all_words_st = []
    all_words = []
    
    print('*****stemmings*****')
    im = len(collection)
    for i,string in enumerate(collection):
        string = _from_string_to_string(string.lower())
        lowered = _lower_stem(string, rule, ps)
        all_words.extend(np.unique(lowered.split(' ')).tolist())
        
        #all_words.append([lowered.split(' ')])
        stemmed_lowered = _lower_stem(lowered, rule_st, ps)
        all_words_st.extend(np.unique(stemmed_lowered.split(' ')).tolist())
        #all_words_st.extend(stemmed_lowered.split(' '))
        marked_collection.append([string, _remove_extra(lowered), _remove_extra(stemmed_lowered)])
        print(f'{i} out of {im}')
    print('*****stemming - done. Creating dictionaries*****')
    mark_dict_st = {}
    mark_dict = {}
    for word in all_words_st:
        try:
            mark_dict_st[word]+=1
        except:
            mark_dict_st[word]=1
    print('*****first dictionary - done*****')
    for word in all_words:
        try:
            mark_dict[word]+=1
        except:
            mark_dict[word]=1
    print('*****second dictionary - done*****')

    
    with open('original_strings.txt', 'w') as file:
        file.write('\n'.join([x[0] for x in marked_collection]))
    with open('lowered_strings.txt', 'w') as file:
        file.write('\n'.join([x[1] for x in marked_collection]))
    with open('lowered_and_stemmed_strings.txt', 'w') as file:
        file.write('\n'.join([x[2] for x in marked_collection]))
        
    with open('mark_dict_st.json', 'w') as outfile:
        json.dump(mark_dict_st, outfile)
        
    with open('mark_dict.json', 'w') as outfile:
        json.dump(mark_dict, outfile)
    
    return marked_collection, mark_dict_st, mark_dict

def _get_bigramms(string):
    #string = _remove_extra(string)
    words = [_remove_extra(x) for x in string.split(' ') if _remove_extra(x)!='']
    
    bigramms0 = np.unique([words[i]+' '+words[i+1] for i in range(len(words)) if i<len(words)-1]).tolist()
    bigramms1 = np.unique([words[i]+words[i+1] for i in range(len(words)) if i<len(words)-1]).tolist()
    
    bigramms = [[bigramms0[i], bigramms1[i]] for i in range(len(bigramms0))]
    #bigramms = np.unique([[x+' '+y, x+y] for x in words for y in words if x!=y]).tolist()
    return bigramms

def _get_symbol_ngramms(string):
    words = ' '+' '.join([_remove_extra(x) for x in string.split(' ') if _remove_extra(x)!=''])+' '
    ngramms = np.unique([words[i:i+3] for i in range(len(words)-3)]).tolist()
    return [[x,x] for x in ngramms]
    
def _sep_words(string):
    words = []
    for x in string.split(' '):
        a = _remove_extra(x)
        if a!='':
            words.append(a)
    words = np.unique(words).tolist()
    
    return [[x,x] for x in words]

def _tf_idf(string, matched_words, marked, N):    
    #tf(t,d) = count of t in d / number of words in d (here we use custom t in d + 1)
    #df(t) = occurrence of t in documents
    #idf(t) = N/df
    #idf(t) = log(N/(df + 1))
    #tf-idf(t, d) = tf(t, d) * log(N/(df + 1))
    
    words = string.split(' ')
    mw = ' '+matched_words[0]+' '
    lmws=len(matched_words[0].split(' '))
    
    a = [(mw.count(' '+word+' ') / lmws)*math.log10(N/(marked[word]+1)) for word in words if ' '+word+' ' in mw]
    return sum(a)#sum_tfidf

def _process_tf_idf(x,ck,key,marked,cl,result):
    try:
        result[key]+=_tf_idf(x,ck,marked,cl)
    except:
        result[key]=_tf_idf(x,ck,marked,cl)


def _search(string, collection, marked,ent_border,cl,st_p):
    #marked - dict with words frequency
    #string - list of bigramms / words
    #collection - list of merged words texts
    
    maxc = len(collection[0])-1
    
    if maxc == 0:
        yi = 0
    else:
        yi=1
    
    #cl = len(collection)
    
    ce = {}
    cy = {}
    maxce = 0
    for i,x in enumerate(collection):
        for j,y in enumerate(string):
            if y[yi] in x[maxc]:
                try:
                    ce[i+st_p]+=1
                    cy[i+st_p].append(j)
                except:
                    ce[i+st_p]=1
                    cy[i+st_p] = [j]
        try:
            if ce[i+st_p]>maxce:
                maxce=ce[i+st_p]
        except:
            pass
    result = {}
    for key,value in ce.items():
        if value>=maxce*ent_border:
            for i in cy[key]:
                try:
                    result[key]+=_tf_idf(string[i][0],collection[key-st_p],marked,cl)
                except:
                    result[key]=_tf_idf(string[i][0],collection[key-st_p],marked,cl)
    return result


def _parallel_f_1(d1,search_request,marked_collection,mark_dict,ent_border,cl,st_p):
    for key,value in _search(_get_bigramms(search_request), [[x[0]] for x in marked_collection], mark_dict,ent_border,cl,st_p).items():
        d1[key]=value
    
def _parallel_f_2(d2,search_request,marked_collection,mark_dict,ent_border,cl,st_p):
    for key,value in _search(_get_bigramms(search_request), [x[0:2] for x in marked_collection], mark_dict,ent_border,cl,st_p).items():
        d2[key]=value
    
def _parallel_f_3(d3,search_request,marked_collection,mark_dict_st, rule, ps,ent_border,cl,st_p):
    for key,value in _search(_get_bigramms(_lower_stem(search_request, rule, ps)),marked_collection, mark_dict_st,ent_border,cl,st_p).items():
        d3[key]=value
    
def _parallel_f_4(d4,search_request,marked_collection,mark_dict,ent_border,cl,st_p):
    for key,value in _search(_sep_words(search_request), [x[0:2] for x in marked_collection], mark_dict,ent_border,cl,st_p).items():
        d4[key]=value
    
def _parallel_f_5(d5,search_request,marked_collection,mark_dict_st, rule, ps,ent_border,cl,st_p):
    for key,value in _search(_sep_words(_lower_stem(search_request, rule, ps)),marked_collection, mark_dict_st,ent_border,cl,st_p).items():
        d5[key]=value
          
def _parallel_f_n1(nd,search_request,marked_collection,mark_dict, weight,ent_border,cl,st_p):
    for key,value in _search(_sep_words(search_request),marked_collection, mark_dict,ent_border,cl,st_p).items():
        try:
            nd[key]+=value*weight
        except:
            nd[key]=value*weight

def _parallel_f_n2(nd,search_request,marked_collection,mark_dict_st, rule, ps, weight,ent_border,cl,st_p):
    for key,value in _search(_sep_words(_lower_stem(search_request, rule, ps)),marked_collection, mark_dict_st,ent_border,cl,st_p).items():
        try:
            nd[key]+=value*weight
        except:
            nd[key]=value*weight
            
def _parallel_f_n(nd,search_request,marked_collection,mark_dict_st, mark_dict, rule, ps, weight,ent_border,cl,st_p):
    p1 = Process(target = _parallel_f_n1, args = (nd,search_request,marked_collection,mark_dict, weight,ent_border,cl,st_p))
    p2 = Process(target = _parallel_f_n2, args = (nd,search_request,marked_collection,mark_dict_st, rule, ps, weight,ent_border,cl,st_p))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
def _part_search(d1_e,d2_e,d3_e,d4_e,d5_e,search_request,marked_collection,mark_dict,mark_dict_st, rule,ps,ent_border,cl,st_p):
    d1i = Manager().dict()
    d2i = Manager().dict()
    d3i = Manager().dict()
    d4i = Manager().dict()
    d5i = Manager().dict()
    
    p1 = Process(target = _parallel_f_1, args=(d1i,search_request,marked_collection,mark_dict,ent_border,cl,st_p))
    p1.start()
    p2 = Process(target = _parallel_f_2, args=(d2i,search_request,marked_collection,mark_dict,ent_border,cl,st_p))
    p2.start()
    p3 = Process(target = _parallel_f_3, args=(d3i,search_request,marked_collection,mark_dict_st, rule, ps,ent_border,cl,st_p))
    p3.start()
    p4 = Process(target = _parallel_f_4, args=(d4i,search_request,marked_collection,mark_dict,ent_border,cl,st_p))
    p4.start()
    p5 = Process(target = _parallel_f_5, args=(d5i,search_request,marked_collection,mark_dict_st, rule, ps,ent_border,cl,st_p))
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    
    for key,value in d1i.items():
        d1_e[key]=value
        
    for key,value in d2i.items():
        d2_e[key]=value
        
    for key,value in d3i.items():
        d3_e[key]=value
        
    for key,value in d4i.items():
        d4_e[key]=value
    
    for key,value in d5i.items():
        d5_e[key]=value
    
def _part_ft_search(nd,new_requests,marked_collection,mark_dict_st,mark_dict, rule, ps,weight,ent_border,cl,st_p):
    ndi = Manager().dict()
    processes = []
    for request in new_requests:
        processes.append(Process(target = _parallel_f_n, args=(ndi,request,marked_collection,mark_dict_st,mark_dict, rule, ps,weight,ent_border,cl,st_p)))
    
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
    
    for key,value in ndi.items():
        nd[key]=value

def search_engine(search_request, collection=None, marked_collection=None, mark_dict_st = None, mark_dict = None, ps = SnowballStemmer("russian"), border_for_pro_search = 30, ent_border = 0.8, ft_model = "ftmod_no_stem.bin"):
    #НУЖНО возвращать ответ по частям
    #НУЖНО параллелить поиск
    #В движок приходит уже marked_collection, скорректированная по дополнительным фильтрам, если такие есть (если 
    #Если не marked_collection то размечается вся коллекция
    
    #разметка коллекции - полный текст, полный текст, приведенные к нижнему регистру и после стемминга (тексты подготовлены с учетом деления символов)
    #словарь - какое слово сколько раз повторяется в коллекции (с учетом стемминга и нижнего регистра)
    #коллекции в БД
    if marked_collection==None and collection!=None:
        print('***create collection***')
        marked_collection, mark_dict_st, mark_dict = _mark_collection(collection, ps)
    elif collection==None and marked_collection==None:
        print('Nowhere to search')
        
    if marked_collection!=None:
        rule = {'stemming': True, 'lowercase': True}
        
        search_request = _from_string_to_string(search_request.lower())
        
        search_transformations = {}
        
        print('***making base search***')
        
        cl = len(marked_collection)

        d1 = Manager().dict()
        d2 = Manager().dict()
        d3 = Manager().dict()
        d4 = Manager().dict()
        d5 = Manager().dict()
        
        n_s_p = 30
        csm = len(marked_collection)
        cs = int(csm/n_s_p)
        if cs>50:
            sub_collections = [marked_collection[i*cs:(i+1)*cs] if i!=n_s_p-1 else marked_collection[i*cs:csm] for i in range(10)]
        else:
            sub_collections = collection
            st_p = 0
        
        st_p = 0
            
        p = []
        for i,sm_c in enumerate(sub_collections):
            st_p = i*cs
            p.append(Process(target = _part_search, args = (d1,d2,d3,d4,d5,search_request,sm_c,mark_dict,mark_dict_st, rule,ps,ent_border,cl,st_p)))
        
        for pr in p:
            pr.start()
            
        for pr in p:
            pr.join()
        
        '''p1 = Process(target = _parallel_f_1, args=(d1,search_request,marked_collection,mark_dict,ent_border,cl,st_p))
        p1.start()
        p2 = Process(target = _parallel_f_2, args=(d2,search_request,marked_collection,mark_dict,ent_border,cl,st_p))
        p2.start()
        p3 = Process(target = _parallel_f_3, args=(d3,search_request,marked_collection,mark_dict_st, rule, ps,ent_border,cl,st_p))
        p3.start()
        p4 = Process(target = _parallel_f_4, args=(d4,search_request,marked_collection,mark_dict,ent_border,cl,st_p))
        p4.start()
        p5 = Process(target = _parallel_f_5, args=(d5,search_request,marked_collection,mark_dict_st, rule, ps,ent_border,cl,st_p))
        p5.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()'''
            
        search_transformations['super_original_bigramms'] = d1
        search_transformations['original_bigramms'] = d2
        search_transformations['stemmed_bigramms'] = d3
        search_transformations['original_words'] = d4
        search_transformations['stemmed_words'] = d5
        
        search_results = {}
        for key,value in search_transformations.items():
            if key == 'super_original_bigramms':
                weight = 3
            elif key == 'original_bigramms':
                weight = 2
            elif key == 'stemmed_bigramms':
                weight = 1
            elif key == 'original_words':
                weight = 0.7
            elif key == 'stemmed_words':
                weight = 0.5
            for subkey, subvalue in value.items():
                try:
                    search_results[subkey]+=subvalue*weight
                except:
                    search_results[subkey]=subvalue*weight
        
        '''
        THIS IS DONE DUE TO FASTTEXT WORKING
        
        not_matched_stemmed_word = _sep_words(_lower_stem(search_request, rule, ps))
        stemmed_bigramms_with_the_word
        get_candidates
        check_distance
        '''
                    
        #print(len(search_results))
        #МОЖНО ВООБЩЕ УБРАТЬ ФАСТТЕКСТ И ПРОБОВАТЬ N-Gramm БУКВ
        #если в тех все плохо - ДОЛЖЕН БЫТЬ ВНЕШНЕ РЕГУЛИРУЕМ
        if len(search_results)<border_for_pro_search:
            print('***making pro search***')
            weight = 3
            model = fasttext.load_model(ft_model)
            r_w = search_request.split(' ')
            new_requests = []
            for w in r_w:
                neig = model.get_nearest_neighbors(w)[0]
                if neig[0]>0.3:
                    new_requests.append(re.sub(' '+w+' ',' '+neig[1]+' ',' '+search_request+' '))

            nd = Manager().dict()
            
            p = []
            for i,sm_c in enumerate(sub_collections):
                st_p = i*cs
                p.append(Process(target = _part_ft_search, args = (nd,new_requests,sm_c,mark_dict_st,mark_dict, rule, ps,weight,ent_border,cl,st_p)))
            
            for pr in p:
                pr.start()
                
            for pr in p:
                pr.join()
            
            '''for request in new_requests:
                processes.append(Process(target = _parallel_f_n, args=(nd,request,marked_collection,mark_dict_st,mark_dict, rule, ps,weight,ent_border,cl,st_p)))
            
            for process in processes:
                process.start()
            
            for process in processes:
                process.join()'''
                
            for key,value in nd.items():
                try:
                    search_results[key]+=value
                except:
                    search_results[key]=value
    return search_results

def _L_distance(a,b):
    #copy-pasted function
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current_row = range(n + 1)  # 0 ряд - просто восходящая последовательность (одни вставки)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
    return current_row[n]

def custom_unique(t_l):
    ps = SnowballStemmer("russian") 
    do_not_return_numbers = []
    n_t_l = []
    for obj in t_l:
        obj = obj.lower()
        string = ''
        for sobj in obj.split():
            sobj = re.sub('[^\w]', '_', sobj)
            if string!='':
                string = string + ' ' + ps.stem(sobj)
            else:
                string = ps.stem(sobj)
        n_t_l.append(string)
    
    for i,x in enumerate(n_t_l): 
        for j,y in enumerate(n_t_l):
            if x in y and i!=j and j not in do_not_return_numbers:
                do_not_return_numbers.append(i)
                break
            if i<j:
                if _L_distance(x,y)<=2:
                    do_not_return_numbers.append(i)
                    break
    
    to_return = []              
    for i,x in enumerate(t_l):
        if i not in do_not_return_numbers:
            to_return.append(x)
                    
    return to_return

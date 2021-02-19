# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:27:59 2021

@author: DZX20
"""
import time
import re
import numpy as np
import string
import en_core_web_sm
nlp=en_core_web_sm.load()
from nltk.tokenize import sent_tokenize, word_tokenize   
import numpy as np 
from gensim.matutils import sparse2full
import gc
import pandas as pd
import gensim
from nltk.stem import PorterStemmer
import nltk 
import logging
stemmer = PorterStemmer()


def data_clean(text):
    text = text.lower() #lower case
    text = re.sub("[^a-zA-Z\s]", '', text)
    doc = nlp(text)    
    text = " ".join([token.lemma_ for token in doc]).strip()
    text=custom_stopwords(text)
    text =dictio_check(text)
    text = stemming(text)
    #text=re.sub(r'please term\b','please terminate',text)
    return text

def stemming(text):
    doc=nlp(text)
    strng=''
    for token in doc:
        strng=strng+" "+ stemmer.stem(token.text)
    return strng.strip()
#search for keyword pron 


def transform_data(data,tfidf_fit,tfidfngram_fit):
    print("data transform started")
    gc.collect()
    test=data.copy()
    # we are appending the Body with the From , To, Subject columns to concatenate one email. 
    test['clean_text'] = test['text'].apply(lambda x: data_clean(x))
    print("clean text is")
    logging.info(test['clean_text'])
    #print(test2['clean_text'])
    
    test_tfidf = tfidf_fit.transform(test['clean_text']).toarray()
    test_tfidfngram = tfidfngram_fit.transform(test['clean_text']).toarray()
    
    # giving names to columns and making a dataframe 
    tfidf_cols = ['tfidf_uni_'+str(i) for i in tfidf_fit.get_feature_names()]
    tfidf_ngram_cols = ['tfidf_ngram_'+str(i) for i in tfidfngram_fit.get_feature_names()]
    
    # converting the embeddings to dataframes 
    
    tfidf_df = pd.DataFrame(test_tfidf,columns = tfidf_cols)
    tfidf_ngram_df = pd.DataFrame(test_tfidfngram, columns = tfidf_ngram_cols)
    
    data_transform = pd.concat([test,tfidf_df,tfidf_ngram_df],axis=1)
    del tfidf_df
    del tfidf_ngram_df
    
    training_cols = tfidf_cols+tfidf_ngram_cols
    c=data_transform[training_cols]
    
    print("data returning")
    del test
    
    del data_transform
    gc.collect()
    return c



def custom_stopwords(text):
    text=re.sub(r'as\b','',text)
    text=re.sub(r'of\b','',text)
    text=re.sub(r'the\b','',text)
    text=re.sub(r'to\b','',text)
    text=re.sub(r'hi\b','',text)
    text=re.sub(r'from\b','',text)
    text=re.sub(r'be\b','',text)
    text=re.sub(r'd i\b','',text)
    
    return text

def dictio_check(text):
    text=re.sub(r'canceled\b','cancelled',text)
    text=re.sub(r'mistanenly\b','mistaken',text)
    text=re.sub(r'cancelation\b','cancellation',text)
    text=re.sub(r'canceling\b','cancelling',text)
    text=re.sub(r'cancelation.\b','cancellation',text)
    text=re.sub(r'canceled\b','cancelled',text)
    text=re.sub(r'canceled\b','cancelled',text)
    text=re.sub(r'cancelations\b','cancellations',text)
    text=re.sub(r'carcel\b','cancel',text)
    text=re.sub(r'cancele\b','cancel',text)
    text=re.sub(r'cancelation\b','cancellation',text)
    text=re.sub(r'cancelations\b','cancellations',text)
    text=re.sub(r'cancell\b','cancel',text)
    text=re.sub(r'cancelation\b','cancellation',text)
    text=re.sub(r'temination\b','termination',text)
    text=re.sub(r'terminatin\b','terminate',text)
    text=re.sub(r'terminaition\b','terminate',text)
    text=re.sub(r'terminaton\b','terminate',text)
    text=re.sub(r'termiante\b','terminate',text)
    text=re.sub(r'termiated\b','terminate',text)
    text=re.sub(r'terminat\b','terminate',text)
    text=re.sub(r'administrator\b','administrator',text)
    text=re.sub(r'adminstrator\b','administrator',text)
    text=re.sub(r'adminsitrators\b','administrator',text)
    text=re.sub(r'adminitrator\b','administrator',text)
    text=re.sub(r'administators\b','administrator',text)
    text=re.sub(r'counseling\b','counselling',text)
    text=re.sub(r'insureds\b','insured',text)
    text=re.sub(r'furloughed\b','furlough',text)
    text=re.sub(r'intendned\b','intended',text)
    text=re.sub(r'totaling\b','totalling',text)
    text=re.sub(r'orthopedic\b','orthopaedic',text)
    text=re.sub(r'atached\b','attached',text)
    text=re.sub(r'pediatrics\b','paediatrics',text)
    text=re.sub(r'terminiation\b','termination',text)
    text=re.sub(r'termianted\b','terminated',text)
    text=re.sub(r'terminaton\b','termination',text)    
    text=re.sub(r'terminatin\b','terminating',text)
    text=re.sub(r'termd\b','termed',text)
    text=re.sub(r'teremed\b','termed',text)
    text=re.sub(r'premiunms\b','premium',text)
    text=re.sub(r'emailcancellation\b','email cancellation',text)
    text=re.sub(r'cance\b','cancel',text)
    text=re.sub(r'distibuting\b','distributing',text)
    text=re.sub(r'discrepencies\b','discrepancies',text)
    text=re.sub(r'resignedlast\b','resigned last',text)
    text=re.sub(r'pediatrics\b','paediatrics',text)
    #text=re.sub(r'pediatrics\b','paediatrics',text)
    
    return text
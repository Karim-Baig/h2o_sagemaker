# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 03:12:38 2020

@author: DZX20
"""
import os
import re
# to import user defined modules
import sys
# import libraries
import numpy as np
import gc
import string
import spacy
from IPython import get_ipython
#nlp = spacy.load('en_core_web_sm') #Load english 
#get_ipython().system('python -m spacy download en_core_web_sm')
#get_ipython().system('pip install autocorrect')
import en_core_web_sm
nlp=en_core_web_sm.load()
from bs4 import BeautifulSoup #only for html tags
#from autocorrect import Speller #Spell check

#import boto3
#import sagemaker
#from sagemaker.amazon.amazon_estimator import get_image_uri
#from sagemaker.session import s3_input, Session 
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
import nltk 
nltk.download('punkt')
import time
stemmer = PorterStemmer()
#check = Speller(lang='en')
#from spellchecker import SpellChecker
#spell = SpellChecker()
from nltk.stem import PorterStemmer
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
pos_dic = {
    'noun' : ['NN','NNS','NNP','NNPS','NOUN','PROPN'],
    'pron' : ['PRP','PRP$','WP','WP$','PRON'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ','AUX','VERB'],
    'adj' :  ['JJ','JJR','JJS','ADJ'],
    'adv' : ['RB','RBR','RBS','WRB','ADV'],
    'det' : ['DET']
}

def pos_check(x):
    doc = nlp(x)
    cnt_noun = 0
    cnt_pron =0
    cnt_verb=0
    cnt_adj=0
    cnt_adv=0
    cnt_det=0
    try:
        for token in doc:
            pos = token.pos_
            if pos in pos_dic['noun']:
                cnt_noun += 1
            if pos in pos_dic['pron']:
                cnt_pron += 1
            if pos in pos_dic['verb']:
                cnt_verb += 1
            if pos in pos_dic['adj']:
                cnt_adj += 1
            if pos in pos_dic['adv']:
                cnt_adv += 1
            if pos in pos_dic['det']:
                cnt_det += 1
    except:
        pass
    return cnt_noun,cnt_pron,cnt_verb,cnt_adj,cnt_adv,cnt_det

def get_log(sta,end):
    return np.log10((sta+1)/(end+1))
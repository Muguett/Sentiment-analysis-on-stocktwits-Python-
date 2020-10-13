#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:23:55 2020

@author: muguet
"""
import pandas as pd
import numpy as np
import pymongo
import nltk
from nltk.corpus import stopwords
import re



#%%

from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt


#%%

client = pymongo.MongoClient("mongodb+srv://username:password@cluster0.qciok.mongodb.net/stoctwits?retryWrites=true&w=majority") 

#Name of the databbase is db 
db = client.StockTwits_WMT

collection=db.tweets



#%%
    


def metatransformation(client, db, query):
    
    text = pd.DataFrame(list(db.tweets.find(query)))
    
    # Compter le nombre de stock 
    
    text["count_stock"] = text["symbols"].apply(lambda x: len(x))
    
    # Extraire un compte unique (on s'interesse uniquement TSL)
    
    text = text[text["count_stock"].isin([1])] 
    
    text["body_transform"] = text["body"].replace(
            #Les expressions régulières, ou plus communément regex (contraction de regular expression) permettent de représenter des modèles de chaînes de caractère.
            regex={
                r"\bnothing\b": "negtag_nothing", 
                #\b : Correspond à la chaîne vide, mais uniquement au début ou à la fin d’un mot. Ex: un mot est défini comme une séquence de « caractères de mots ». r'\bfoo\b' validera 'foo', 'foo.', '(foo)' ou 'bar foo baz' mais pas 'foobar' ou 'foo3'. 
                r"\bno\b": "negtag_no",
                r"\bnone\b": "negtag_none",
                r"\bneither\b": "negtag_neither",
                r"\bnever\b": "negtag_never",
                r"\bnobody\b": "negtag_nobody",
                r"\d+": "numbertag ", #Correspond à "Integer number"
                r"([@?])(\w+)\b": "user", 
                # [] : Utilisé pour indiquer un ensemble de caractères. 
                #(?...) : Il s’agit d’une notation pour les extensions
                #\w : Pour les motifs Unicode (str) : Valide les caractères Unicode de mot ; cela inclut la plupart des caractères qui peuvent être compris dans un mot d’une quelconque langue, aussi bien que les nombres et les tirets bas
                r"\b&#\b": " ",
                # Supprimer l'unicode 
                r"[$][A-Za-z][\S]*": "",
                # Supprimer le ticker ($)
                #r"\W": " ", # JE NE VEUX PAS SUPPRIMER LES EMOJIS :(
                #\W": Valide tout caractère qui n’est pas un caractère de mot. (Supprimer tout caractère qui n’est pas un caractère de mot.)
                r"\s+[a-zA-Z]\s+": " ",
                #\s: Pour les motifs Unicode (str) :Valide les caractères d’espacement Unicode (qui incluent [ \t\n\r\f\v] et bien d’autres, comme les espaces insécables requises par les règles typographiques de beaucoup de langues)
                r"\^[a-zA-Z]\s+": " ",
                #^[a-zA-Z] means any a-z or A-Z at the start of a line
                r"\s+": " ",
                r"^b\s+": "",
                r"\bTSLA\b": "",
                # Supprimer TSLA
                r"\bTesla\b": "",
                # Supprimer tesla
                r"\btesla\b": "",
                 # Supprimer http
                r"https\b": "",
                 # Supprimer tesla
                r"\bcom\b": "",
                # Supprimer www
                r"\bwww\b": "",
                # Supprimer website body
                r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))": "url"
    
    
            }
        )
    
    emoji_pat = '[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]'
    shrink_whitespace_reg = re.compile(r'\s{2,}')
    
    def clean_text(raw_text): #remove everything except words and emoji from text

        reg = re.compile(r'({})|[^a-zA-Z]'.format(emoji_pat)) # line a
        result = reg.sub(lambda x: ' {} '.format(x.group(1)) if x.group(1) else ' ', raw_text)
        return shrink_whitespace_reg.sub(' ', result)

    text["body_transform"]=  text.body_transform.apply(clean_text)

    # Lower
    
    text["body_transform"] = text["body_transform"].str.lower() #Python string method lower() returns a copy of the string in which all case-based characters have been lowercased.
        
    
        # Remove stop words
    
    stop = stopwords.words('english')
    text["body_transform"] = text["body_transform"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
        # Lemmatize
    
    twitter_tokenizer = nltk.tokenize.TweetTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = SnowballStemmer("english")   
    
    def lemmatize_text(text):
        return [stemmer.stem(lemmatizer.lemmatize(w)) for w in twitter_tokenizer.tokenize(text)]
    
    text['tokenized_clean_text'] = text.body_transform.apply(lemmatize_text)
    

    return text

query ={"sentiment":{ "$ne": "Neutral" }} #This query will select all documents in the inventory collection where the qty field value does not equal x

CleanDataBase=  metatransformation(client, db, query)

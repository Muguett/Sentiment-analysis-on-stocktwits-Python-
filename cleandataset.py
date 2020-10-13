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



#%%VADER


from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#vader function adds a new column which shows the component vader sentiment value of a given stock
def vadermet(df,nameoftokenizedtext):
    df4=df.reset_index(drop=True)
    y=0
    emptl=[]
    for list in df[str(nameoftokenizedtext)]:
        y= TreebankWordDetokenizer().detokenize(list)
        emptl.append(y)
    
    sid = SentimentIntensityAnalyzer()
    varcomplist=[]
    ss=0
    for sentence in emptl:
        ss = sid.polarity_scores(sentence)
        varcomplist.append(ss)
    
    df2 = pd.DataFrame(varcomplist) 
   
    df3 = df2["compound"]
    
    VADER_SENT=[]
    s=0
    
    for i in df3:
        if i>0:
            s= 1.00
            VADER_SENT.append(s)
        if i<0:
            s= -1.00
            VADER_SENT.append(s)
        if i==0:
            s= 0.00
            VADER_SENT.append(s)
            
    df5=pd.DataFrame(VADER_SENT)
    df5.columns=["Vader Sentiment"]    
    
    result = pd.concat([df4, df3, df5], axis=1,ignore_index=False)

    return result 


result = vadermet(CleanDataBase, "tokenized_clean_text")


 

#%%
# ## Get Walmart Stock Returns Data

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas_datareader.data import get_data_yahoo


enddate=datetime(2020,10,5)
startdate=datetime(2009,9,2)
stock_data = get_data_yahoo('WMT',startdate,enddate)
stock_data['Volume'].plot(legend=True,figsize=(10,4));
stock_data.head()
stock_data['Adj Close'].plot(legend=True,figsize=(10,4))

stock_data['returns']= stock_data['Adj Close'].pct_change(1) #returns

del client, enddate, query, startdate, stopwords

#%% Group tweets with same date and compute avarage sentiment

result['created_at'] = pd.to_datetime(result['created_at'])

y=result.groupby(pd.Grouper(key='created_at', freq='D')).mean()

stock_data["date"]=pd.to_datetime(stock_data.index)

stock_data2= stock_data.groupby(pd.Grouper(key='date', freq='D')).mean() #we already know that we have unique return for each day, I used this fonction to create a nan value for the missed dates (we have missed days because the stock market is not open every day )

#%% 
WalmartReturns= pd.DataFrame(stock_data2["returns"])    
WalmartVaderSentiment= pd.DataFrame(y["Vader Sentiment"])

#%% Create a data frame with stock return and vader sentiment

#Convert pandas timezone-aware DateTimeIndex to naive timestamp

WalmartVaderSentiment.index= pd.DatetimeIndex([i.replace(tzinfo=None) for i in WalmartVaderSentiment.index])

#merge dataframes

wmt = pd.concat([WalmartVaderSentiment, WalmartReturns],axis=1, join="inner")

wmt.head()



#%%Relationship between sentiments and Walmart return'

axw = wmt[['Vader Sentiment', 'returns']].plot(secondary_y = 'returns', title='Relationship between sentiments and Walmart return')
figw = axw.get_figure()
figw


#%%
from statsmodels.tsa.stattools import grangercausalitytests

#Drop rows with nan values
#Because we can not implement granger causalitytest on values contains NaN or inf 

wmt.dropna(subset = ['Vader Sentiment', 'returns'], inplace=True)

import statsmodels
from statsmodels.tsa.stattools import adfuller

statsmodels.tsa.stattools.adfuller(wmt.returns, regression = 'ct')
#non unit root

# According to the results, we believe there is no granger causality from sentiment to returns, ie all p-values are all above .05, accepting the null hypothesis, the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1.

grangercausalitytests(wmt[['returns', 'Vader Sentiment']], maxlag=4)

# ## Regress
# 
# $$
# r_{i, t}=\alpha+\beta_{1} \Delta s_{1, t}+\beta_{2} \Delta s_{i, t-1}+\epsilon_{t}
# $$
# 

wmt['sentiment_lag'] = wmt['Vader Sentiment'].shift(1)

wmt['L_s1'] = wmt['Vader Sentiment'].pct_change(1)
wmt['L_s2'] = wmt['sentiment_lag'].pct_change(1)
wmt.head()

import statsmodels.formula.api as smf

wmt= wmt.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

mod1 = smf.ols(formula='returns ~ L_s1 + L_s2', data=wmt).fit()
mod1.summary()










                                        

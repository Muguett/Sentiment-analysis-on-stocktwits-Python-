#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:02:20 2020

@author: 
"""

#Step by step:
#1 - Extract data from StockTwits: https://api.stocktwits.com/developers/docs
#2 – Store the data on MongoDB

#####Extract at least 100.000 messages about this stock on StockTwits and store them on a MongoDB database

import urllib3 
import json 
import pymongo
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

client = pymongo.MongoClient("mongodb+srv://username:password@cluster0.qciok.mongodb.net/stoctwits?retryWrites=true&w=majority") 

#Name of the databbase is db 
db = client.StockTwits_WMT

#Collection name
collection_name=db.tweetst

http = urllib3.PoolManager()

ticker= "WMT"

warnings.filterwarnings('ignore')

def getdata(database,collection,ticker):
   
    try:
        url="https://api.stocktwits.com/api/2/streams/symbol/{0}.json?max={1}".format(ticker,lastid)
    except:
        url="https://api.stocktwits.com/api/2/streams/symbol/{0}.json".format(ticker)
    
    data = json.loads(http.request('GET', url).data)
    print(url)
    if data["response"]["status"] == 200: #if HTTP response status of http data is "Successful responses":
        
        for i, element in enumerate(data["messages"]):
            
            id_tweet=element["id"]
            text_tweet=element['body']
            username=element['user']['username']
            date= element['created_at']
                
            sent_var = element['entities']['sentiment']
                
            if sent_var:
                sent_var_type=sent_var["basic"] #extract only stocktweets with basic sentiments
            else:
                sent_var_type = "Neutral"
                
            stocktwits_symbols=[] 
            for symbol in element["symbols"]:
                stocktwits_symbols.append(symbol["title"])
            
            try:
                collection_name.insert_one({"id":id_tweet,"created_at":date,"user":username,"sentiment":sent_var_type,"body":text_tweet,"symbols":stocktwits_symbols})
            except:
                pass
                
            if i==29:
                latest_id=element["id"]
                latesttweetdic= {"id": latest_id,"url": url}
                return latesttweetdic
    
    else:
        status= 429 #The user has sent too many requests in a given amount of time ("rate limiting").
        return status
   

# The first 30 messages 
result = getdata(db,
                 collection_name,
                 ticker)

#This line of code insures that each document in the dataset has a unique id 
db["tweetst"].create_index( "id", unique= True )

# Loop to get the messages
# StockTwits permits only  200 unauthenticated calls per hour 
#(If your application is being rate-limited it will receive HTTP 429 response code)
# This loop makes 200 requests per. 
# The loop permits us to get historical StockTwits data on a given stock. 
    
timesecond = 0

for k in range(0,8):
    time.sleep(timesecond)
    begin = datetime.now()
    for i in tqdm(range(0, 250)):
        ids=[]
        for element in db["tweets"].find({},limit=1).sort("id",pymongo.ASCENDING):
            ids.append(element["id"])
        lastid =str(min(ids))

        result = getdata(
                db,
                collection_name,
                ticker)
   
        if result == 429:
            end = datetime.now()
            time_code = end - begin
    
            time_next_batch = begin + timedelta(hours=1)
            time_end_batch = begin + timedelta(seconds=time_code.seconds)
            timesecond = (time_next_batch - time_end_batch).seconds
            
            break
            
        else:
            
            lastid = result['id']
    
    print('Next batch in {} minutes. It will happen at {}'.format(timesecond/60, 
                                           time_next_batch.strftime("%H:%M:%S"))
         )

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import math
import requests
import time


# Generates list of all asset ids to pull from api
csvFile = open("biglist.csv", 'r')
reader = csv.reader(csvFile)
assetIDs = list(reader)
print "asset ids opened."
print assetIDs[2]

#URL preparation
url1 = 'http://128.82.5.11:8082/tseries?assetid='
url2 = '&atype=Story'

print "Going to API"

# editing the url string text
count = 0
for ID in assetIDs:
    url_x = str(ID)
    url_x = url_x.replace('\n','')
    url_x = url_x.replace("'","")
    url_x = url_x.replace("[","")
    url_x = url_x.replace("]","")
    url_T = url1 + url_x + url2
    print url_T
    request = requests.get(url_T)
    print "Article Pull Count: ", count
    print "Current ID: ", ID
    if request.status_code == 200:
        #data pull from the api
        data = pd.read_json(url_T)
        f_data = pd.DataFrame(data)
        if f_data.values.max() > 15:
			print ID
        #create csv file name
        file_name_csv = url_x + ".csv"
        f_data.to_csv(file_name_csv, sep='\t', encoding='utf-8')
    else:
		print("Asset API Error ", url_x)
    print "sleeping"
    count = count + 1
    time.sleep(15)
     
      

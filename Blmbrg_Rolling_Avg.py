
# coding: utf-8

# In[57]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_json('http://128.82.5.11:8082/tseries?assetid=O2RGJ0SYF01X01&atype=Story')
f_data = pd.DataFrame(data)



# In[36]:




# In[67]:

f_data.plot(figsize=(8,5))
plt.show()


# In[68]:

import math
f_data['100Avg'] = pd.rolling_mean(f_data, window=100)
f_data['200Avg'] = pd.rolling_mean(f_data['series'], window=200)
#f_data['Mov_Vol'] = pd.rolling_std(f_data['series'], window=150)*math.sqrt(200)
print f_data.info()


# In[70]:

f_data[['series', '100Avg', '200Avg']].plot(figsize=(8,5))

plt.show()


# In[ ]:




# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:01:46 2018

@author: Sunny
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:38:59 2018

@author: Sunny
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:15:15 2018

@author: Sunny
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:45:25 2018

@author: Sunny
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:02:51 2018

@author: Sunny
"""

    








import os
os.environ["KERAS_BACKEND"]="theano"
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np

import random

# Python code to remove duplicate elements
def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

  


def randbin1(d):
    mx=(2**d)-1
    b=bin(random.randint(0,mx))
    return b[2:].rjust(d,'0')



    
    


  

#from matplotlib import pyplot
file= 'dataset_1_fp.xlsx'
xl=pd.ExcelFile(file) #importing the excel file
#print(xl.sheet_names)
df1=xl.parse('Sheet1')
df1=df1[df1.columns[1:7]]
values=df1.values
values = values.astype('float32')
attr=values.shape[1]-1 #attr attributes are there in the loaded dataset

def crossover(df):     #choose any 2 strings from the population
    [p,q]=random.sample(range(0,len(df)),2)     #number of elements in the list
    [ch1,ch2]=random.sample(range(0,attr),2) #number of atrributes
    print(p,q,ch1,ch2)
    ele1=df[p]
    ele2=df[q]
    #print(ele1,ele2)
    ele1=list(ele1)
    ele2=list(ele2)
    x=ch1 #index of crossover
    y=ch2
    (ele1[x], ele2[x])=(ele2[x], ele1[x])
    (ele1[y], ele2[y])=(ele2[y], ele1[y])
    ele1="".join(map(str, ele1))
    ele2="".join(map(str, ele2))
    retlist=[]
    retlist.append(ele1)
    retlist.append(ele2)
    return retlist
    
    
def mutation(st):
    
    index=random.sample(range(0,attr),1)
    ele1=st
    
    
    ele1=list(ele1)
    
    if ele1[index[0]]=='1':
        ele1[index[0]]='0'
    else:
        ele1[index[0]]='1'
    
    
    ele1="".join(map(str, ele1))
    retlist=[]
    retlist.append(ele1)
    return retlist
    

new1="1"*attr
new0="0"*attr  
i=0;
populn=[]
popln1=[]
init=[]
for x in range(100):
    x=x+1
    populn.append(randbin1(attr))    #define initial collection
    
popln1=list(populn)
print(popln1)
popln1=Remove(populn) #Remove is not pre defined
print(popln1)
if new1 in popln1:
    popln1.remove(new1)

if new0 in popln1:
    popln1.remove(new0)
print(popln1)  #print  collection
popln2=popln1
#initialisation
    
    
init=random.sample(range(0,len(popln1)),10)
popln1=[]
for var in init:
    popln1.append(popln2[var])
popln1=Remove(populn)

if new1 in popln1:
    popln1.remove(new1)

if new0 in popln1:
    popln1.remove(new0)




popln2=popln1
print(popln1) #print population
#scaler =MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
     n_vars = 1 if type(data) is list else data.shape[1]
     df = DataFrame(data)
     cols, names = list(), list()
    # input sequence (t-n, ... t-1)
     for i in range(n_in, 0, -1):
         cols.append(df.shift(i))
         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
     for i in range(0, n_out):
          cols.append(df.shift(-i))
          if i == 0:
              names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
          else:
              names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
             
          agg = concat(cols, axis=1)
          agg.columns = names
          if dropnan:
                    agg.dropna(inplace=True)
          return agg
 #function to list all the occurences of an element         
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs
  
    
#function to search for a key by entering a value
def srchkey(dict_temp,search_val):
    for k,val in dict_temp.items():
        if val== search_val:
            return(k)


#returns list of zeros
def zerolistmaker(n):
   
    return [0]*n






fin_str=zerolistmaker(attr+1) #to store the final inputs for every output 
stor=[] #list to store fitness values
fin_wt=zerolistmaker(attr+1)#final weights
min_error=zerolistmaker(attr+1) #for rmse
cnt=0 #for the 3rd loop
count=0 #for the 2 nd loop
ct=0 #for the first loop
a=[]
ls1=[] #to store the dictionary 1 values
lt1=[] #to store the dictionary 2 values
l2=[]
count1=0
stkeys=[] #to store the dictionary keys 
df1=np.asarray(df1[df1.columns[1:7]]) #converting it to an array
data = series_to_supervised(df1,7,1) #series to supervised
print(data)  
n_train_hrs=6 #total time/tuples for training
train = values[:n_train_hrs, :]
test = values[n_train_hrs:, :]
train1=train
test1=test
crossover_res=[]
mutate_res=[]
dict_1={} #dictionary 1 to store strings and errors
dict_2={} #dictionary 2 to store errors and weights
# split into input and outputs
for ct in range(train.shape[1]):
    train=np.delete(train1,ct,axis = 1) 
    test=np.delete(test1,ct,axis = 1)
    dict_1.clear()
    dict_2.clear()
    popln1=popln2
    count=0
    choice=input("Enter your choice\n")
    if choice=='n':
        break
    while (choice!='n'):
        
        count=count+1
       
        print(popln1)
        for var in popln1:
            cnt=cnt+1
            print(var)
            a=list(var)
            print(a)
            b=(list_duplicates_of(a,'1'))
            print(b)
            train_X, train_y = train[:,b], train1[:,ct]
            test_X, test_y = test[:,b], test1[:,ct]
            train_X = train_X.reshape((n_train_hrs,1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0],1, test_X.shape[1]))
            print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')
            wmat=model.get_weights()
            #print(wmat)
            history = model.fit(train_X, train_y, epochs=1, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    #make a prediction
            yhat = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # calculate RMSE
            # print(train_y[4:],yhat)
            rmse = sqrt(mean_squared_error(train_y[4:], yhat))
            dict_1[var]=rmse
            dict_2[rmse]=wmat
            model.pop()
         
        print(dict_1)
        print(dict_2)
        popln3=popln1 #storing the initial population    
        ls1=[]
        lt1=[]
        stkeys=[]
        ls1=list((dict_1.values()))
        
        ls1.sort()
        
        
        ls1=ls1[0:4] #top 4 values for selection
        
       # print(ls1)
        for val in ls1:
             lt1.append(dict_2[val])            #choose the corresponding weight matrices
        for val in (ls1):
            stkeys.append(srchkey(dict_1,val))
        
           
        
        #print(min_error[ct])
        if count==1:
            min_error[ct]=(ls1[0])
            fin_str[ct]=(stkeys[0])
            fin_wt[ct]=(lt1[0])
        else:
            if ls1[0]<min_error[ct]:
                min_error[ct]=ls1[0]
                fin_str[ct]=stkeys[0]
                fin_wt[ct]=lt1[0]
            
        
        #print(ls1)
        crossover_res=crossover(stkeys)
        crossover_res=crossover(crossover_res) #crossover twice for more variety
        if new1 in crossover_res:
            crossover_res.remove(new1)

        if new0 in crossover_res:
            crossover_res.remove(new0)
            
        #num_mutate=random.sample(range(0,len(stkeys)),1)
        #str_mutate_num=random.sample(range(0,len(stkeys)),num_mutate)
        #mut_coll
        for val in crossover_res:
            mutate_res.append(mutation(val))
            
        if new1 in mutate_res:
            mutate_res.remove(new1)

        if new0 in mutate_res:
            mutate_res.remove(new0)
            
        if mutate_res==[]:
            popln1=crossover_res
        else:
            popln1=mutate_res
        
        
        count1=0
        for var in popln1:
            count1=count1+1
            str1="".join(var)
            popln1[count1-1]=str1
            
        
        print("popln1 is \n")
        print(popln1)    
        dict_1.clear()
        dict_2.clear()
        choice=input("Enter your choice\n")
        
    
    print("the string is %s" % fin_str[ct])
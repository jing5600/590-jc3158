
#################################################
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 23:02:54 2021

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from   scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt


class Data():
    
    def __init__(self):
        return
    
    def read(self,file):
        df = pd.read_json(file)
        return df
    
    def age18(self,df):
        df = df[df['x']<18]
        return df
    

    
    def splitdata(self,df):
        x_train, x_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test  
    
    def vis(self,x,y1,y2):
        plt.scatter(x,y1)
        plt.plot(x,y2,color = 'red')
        plt.show()
    ## normalize
    def norm(self,x_train,y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        x_norm = list(i for i in range(len(x_train)))
        y_norm = list(i for i in range(len(y_train)))
        for i in range(len(x_train)):
            x_norm[i] = (x_train[i]-x_mean)/np.std(x_train)
        for i in range(len(y_train)):
            y_norm[i] = (y_train[i]-y_mean)/np.std(y_train)
        return x_norm , y_norm
    
    
    def linear(self,lst):
        m = lst[0]
        b = lst[1]
        res1 = 0
        for i in range(len(x_train)):
            res1 += (y_norm[i]-(m*x_norm[i]+b))**2
        return res1
        
    ## predict and unnormalize
    def pred(self,x_norm,y_train):
        y_pred = list(i for i in range(len(x_norm)))
        for i in range(len(x_norm)):
            y_pred[i] = (popt1[0]*x_norm[i] + popt1[1])*np.std(y_train)+np.mean(y_train)
        return y_pred
    
    
regression=Data()

## read in the data
df = regression.read('weight.json')

df = regression.age18(df)
## split the data
x_train, x_test, y_train, y_test  = regression.splitdata(df)


x_train = x_train.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)




## normalize
x_norm , y_norm  =  regression.norm(x_train, y_train)



## get the para
lst = [0,0]
res = minimize(regression.linear,lst,method='Nelder-Mead', tol=1e-5)
popt1 = res.x


y_pred = regression.pred(x_norm,y_train)
    
##vis

regression.vis(x_train,y_train,y_pred)





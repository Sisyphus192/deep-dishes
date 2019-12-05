import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import *
from collections import Counter
class tools(object):
    def __init__(self):
        self.df = pd.DataFrame()
        self.tag = None 
        self.word = None 
        self.value = None 
        self.column = None 
    def setdf(self,df):
        self.df = df  
    def settags(self,tag):
        self.tag = tag;
    def setSearchWord(self,word):
        self.word = word 
    def setSearchValue(self,value):
        self.value = value
    def DeleteColumns(self,columns):
        self.column = columns
    def readfile(self,filepath):
        self.df = pd.read_hdf('../data/processed/epi_vector.h5')
    def search(self,df):
        # if(self.tag is None):
        #     temp_df = df[[self.searchword in x for x in df.tags]]
        # else:
        temp_df = df[[self.word in x for x in df.tags]]
        return temp_df
    def filterDFWithCondition(self):
        temp_df = self.df[self.df[self.tag] == self.searchvalue]
        return temp_df
    def DeleteColumns(self,columns):
        self.df = self.df.drop(columns=columns,axis=0)
        return self.df
    def LabelEncoderforDF(self,X,Y):
        le = preprocessing.LabelEncoder()
        temp_x = X.apply(le.fit_transform)
        temp_y = le.fit_transform(Y)
        return temp_x,temp_y
    def countColumnValues(self,df):
        c = Counter()  
        df[self.tag].apply(lambda x: c.update(x))
        return c
if __name__ == "__main__":
    tl = tools()
    tl.readfile('../data/processed/epi_vector.h5')
    print(tl.df.head(1))
    tl.settags("tags")
    tl.countColumnValues(tl.df)
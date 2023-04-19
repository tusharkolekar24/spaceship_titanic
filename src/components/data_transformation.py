import pandas as pd
import numpy as np
import sys
import os
from src.logger import logging
from src.exception import CustomException

class DataTransformation:
        def __init__(self,dataset):
                self.__dataset = dataset

        def missing_value_imputation(self,dataset):
            # import the KNNimputer class
            from sklearn.impute import KNNImputer
            
            #filling missing values
            dataset['HomePlanet'].fillna(dataset['HomePlanet'].mode()[0],inplace=True)
            dataset['Destination'].fillna(dataset['Destination'].mode()[0],inplace=True)
            dataset['CryoSleep'].fillna(dataset['CryoSleep'].mode()[0],inplace=True)
            dataset['VIP'].fillna(dataset['VIP'].mode()[0],inplace=True)
            
            dataset[['Deck','Num','Side']] = dataset['Cabin'].str.split("/",expand=True)
            dataset['Deck'].fillna(dataset['Deck'].mode().values[0],inplace=True)
            dataset['Side'].fillna(dataset['Side'].mode().values[0],inplace=True)
            
            # create an object for KNNImputer
            imputer = KNNImputer(n_neighbors=2)
            after_imputation = imputer.fit_transform(dataset[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Num']])
            After_imputation = pd.DataFrame(after_imputation,columns=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Num'])
            for cols in After_imputation.columns:
                dataset[cols] = After_imputation[cols].values

            return dataset

        def get_scaled_dataset(self,dataset):
            categories_feature = ['Destination','Deck']
            categorical_table  = pd.concat([pd.get_dummies(dataset[feature],columns='{}'.format(feature)) 
                                            for feature in categories_feature],axis=1)
            
            categorical_table['VIP'] = dataset['VIP'].map({True:1,False:0})
            categorical_table['CryoSleep']  = dataset['CryoSleep'].map({True:1,False:0})
            categorical_table['HomePlanet'] = dataset['HomePlanet'].map({'Earth':1,'Europa':0,'Mars':2})
            categorical_table['Side']= dataset['Side'].map({'S':0, 'P':1})
            
            numerical_feature  = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Num'] #,'group','member_group'
            from scipy.special import boxcox1p
            numerical_table = pd.DataFrame(columns=numerical_feature)

            for features in numerical_feature:
                numerical_table[features] = boxcox1p(dataset[features],0.15)
            
            numerical_table['total_expense'] = numerical_table[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
            
            return pd.concat([categorical_table,numerical_table],axis=1)       

        def initiate_data_transformation(self):

            removing_missing_values = self.missing_value_imputation(self.__dataset) 
            
            transform_dataset       = self.get_scaled_dataset(removing_missing_values)

            transform_dataset       = transform_dataset.astype(float)
           
            return transform_dataset
        
# if __name__=='__main__':
#      dataset = pd.read_csv(r'D:\git_practice\spaceship_titanic\original_data\train.csv')
#      object = DataTransformation(dataset)
#      clean_dataset = object.initiate_data_transformation()
#      print(clean_dataset.shape)

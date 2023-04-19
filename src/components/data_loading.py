import pandas as pd
import numpy as np
import sys
import os
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
@dataclass

class DataLoadingConfig:
      scale_train_file_path = os.path.join(os.getcwd(),'model_artifacts','scale_dataset','scaled_trainset.csv')
      scale_test_file_path  = os.path.join(os.getcwd(),'model_artifacts','scale_dataset','scaled_testset.csv')
      
      trainset_file_path = os.path.join(os.getcwd(),'model_artifacts','train_test_split','trainset.csv')
      testset_file_path  = os.path.join(os.getcwd(),'model_artifacts','train_test_split','testset.csv')

class DataLoading:
      def __init__(self):
            self.file_paths = DataLoadingConfig()

      def initiate_data_loading(self):
            logging.info("Data Loading Process Started")
            try:
                scaled_trainset = pd.read_csv(self.file_paths.scale_train_file_path)
                logging.info("Scaled Dataset is initialized")
                
                X_train, X_test = train_test_split(scaled_trainset,
                                                   test_size=0.35)
                
                logging.info("Train dataset split into train/test dataset")

                X_train.to_csv(self.file_paths.trainset_file_path,index=False)
                logging.info("Train dataset are used for model training")

                X_test.to_csv(self.file_paths.testset_file_path,index=False)
                logging.info("Test dataset are used for model training")

            except Exception as e:
                   raise CustomException(e,sys)
            
if __name__=='__main__':
      object = DataLoading()
      object.initiate_data_loading()

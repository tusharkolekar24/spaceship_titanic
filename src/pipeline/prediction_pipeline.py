import pickle
import pandas as pd
import os
from sklearn.metrics import classification_report
from src.utils import get_confusion_matrix,performance_evaluation
import warnings
warnings.filterwarnings('ignore')
from src.exception import CustomException
from src.logger import logging
import sys

class ModelPredictionConfig:
      file_path_trainset = os.path.join(os.getcwd(),'model_artifacts','train_test_split','trainset.csv')
      file_path_testset  = os.path.join(os.getcwd(),'model_artifacts','train_test_split','testset.csv')
      file_path_model_artifacts = os.path.join(os.getcwd(),'model_performance','train_models')
      file_path_train_performance = os.path.join(os.getcwd(),'model_performance','model_evaluation','train')
      file_path_test_performance  = os.path.join(os.getcwd(),'model_performance','model_evaluation','test')

class ModelPredictionPipeline:
      def __init__(self):
           self.file_paths = ModelPredictionConfig()

      def initiate_modelprediction_pipeline(self):
           logging.info("Initiate Model Prediction Pipeline")
           try: 
                cols_name = ['Models','Accuracy','Precision','Recall','F1_score','ROC_AUC_score','Methods']
                trainset_summary = [] # pd.DataFrame(columns=cols_name)
                testset_summary  = [] # pd.DataFrame(columns=cols_name)

                train = pd.read_csv(self.file_paths.file_path_trainset)
                test  = pd.read_csv(self.file_paths.file_path_testset)
                logging.info("Train & Test Dataset loaded in Prediction Pipelines")

                X_train = train.iloc[:,:-1]
                y_train = train.iloc[:,-1]

                X_test = test.iloc[:,:-1]
                y_test = test.iloc[:,-1]

                logging.info("Dataset split into train/test & ready for performing predictions")

                for mode_name in os.listdir(self.file_paths.file_path_model_artifacts):

                    logging.info("{} model select for performing predictions".format(mode_name.split(".pkl")[0]))
                    model = pickle.load(open(os.path.join(self.file_paths.file_path_model_artifacts,mode_name),'rb'))
                    
                    y_pred_test  = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                    get_confusion_matrix(y_train,y_pred_train,
                                        self.file_paths.file_path_train_performance,mode_name.split('.pkl')[0],
                                        'train')

                    get_confusion_matrix(y_test,y_pred_test,
                                        self.file_paths.file_path_test_performance,mode_name.split('.pkl')[0],
                                        'test')                      
                   
                    logging.info("Confusion Matrix Reports generated for {} model".format(mode_name.split(".pkl")[0]))

                    result_summary_trainset = performance_evaluation(y_train,y_pred_train,
                                                                    
                                                                    mode_name.split('.pkl')[0],
                                                                    'train')

                    result_summary_testset  = performance_evaluation(y_test,y_pred_test,
                                                                     
                                                                     mode_name.split('.pkl')[0],
                                                                    'test')
                    
                    logging.info("{} model Performance Evaluated & generated Reports".format(mode_name.split(".pkl")[0]))

                    print(mode_name,'\n',classification_report(y_train,y_pred_train),classification_report(y_test,y_pred_test),'\n')

                    trainset_summary.append([vals for vals in result_summary_trainset.values()])
                    testset_summary.append([vals for vals in result_summary_testset.values()])

                result_summary_train = pd.DataFrame(trainset_summary,columns= cols_name)    
                result_summary_test  = pd.DataFrame(testset_summary,columns= cols_name)
                
                result_summary_train.to_csv(os.path.join(os.getcwd(),'output','train_model_summary.csv'),index=False)
                result_summary_test.to_csv(os.path.join(os.getcwd(),'output','test_model_summary.csv'),index=False)

                return result_summary_train,result_summary_test
           
           except Exception as e:
                 raise CustomException(e,sys)

if __name__=='__main__':
     obj = ModelPredictionPipeline()
     result_summary_train,result_summary_test =  obj.initiate_modelprediction_pipeline()
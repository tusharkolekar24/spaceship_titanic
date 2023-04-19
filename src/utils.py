import pickle
from src.logger import logging
from src.exception import CustomException
import os
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
@dataclass

class FilePathConfig:
      model_save_loc = os.path.join(os.getcwd(),'model_performance','train_models')
      train_model_file_paths = os.path.join(os.getcwd(),'model_performance','evaluating_performance','train')
      test_model_file_paths  = os.path.join(os.getcwd(),'model_performance','evaluating_performance','test')

def Save_Object(model,model_name):
        # print(model_name)
        with open (os.path.join(os.getcwd(),'model_performance','train_models',f'{model_name.replace(" ","_")}.pkl'),'wb') as files:
            pickle.dump(model,files)    
      
def get_confusion_matrix(true,pred,file_path,model_name,types):
    import matplotlib.pyplot as plt
    import seaborn as sns

    #plt.style.use('seaborn')
    fig = plt.figure(figsize=(4,3))  
    cm = confusion_matrix(true,pred)
    sns.heatmap(cm,annot=True,cmap="Blues",
                fmt='g',xticklabels=['Spam','Ham'],
                yticklabels=['spam','ham'])
    plt.title("Confusion Matrics Statistics")
    
    fig.savefig(os.path.join(file_path,f"{model_name}_{types}.jpg"))


def performance_evaluation(true,pred,model_name,type):
      return {'Models':model_name,
              'Accuracy':accuracy_score(true,pred),
              'Precision':precision_score(true,pred),
              'Recall':recall_score(true,pred),
              'F1_score':f1_score(true,pred),
              'ROC_AUC_score':roc_auc_score(true,pred),
              'Methods':type}
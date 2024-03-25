#######################################
############## CODE INIT #############
#######################################

import sys
data_path = sys.argv[1]
print(f'*** ENSEMBLE COMPUTATION ON DATASET {data_path} ')
import glob

# to import temberture code 
sys.path.insert(0, '/ibmm_data/TemBERTure/model/code/') 

#######################################
########## 1. Models loading ##########
#######################################

# Paths Setup: It defines main_path as the root directory for the data and partial_epoch_path as a subdirectory.
main_path = '/ibmm_data/TemBERTure/model'
partial_epoch_path ='/BERTSequential/2_LAYERS_HEAD/PARTIAL_EPOCH_CLS_WEIGHTS/'
full_cls_epoch_path = '/BERTSequential/2_LAYERS_HEAD/FULL_CLS_TRAINING'
random_individual_path = '/BERT_regr/2_LAYER_HEAD'

# Categories: Lists of category names (categories) and composed paths (categories_path) are set.
## *** KEEP THE SAME ORDER BETWEEN categories AND categories_path TO MAKE THE PATH READER WORKING LATER ***
categories = ['0_25', '0_50', '0_75', '1/1E-3','FULL_CLS','RANDOM']
# composed path for each category
categories_path=[f'{partial_epoch_path}0_25',f'{partial_epoch_path}0_50',f'{partial_epoch_path}0_75',f'{partial_epoch_path}1/1E-3',f'{full_cls_epoch_path}',f'{random_individual_path}']
categories_path

from temBERTure.temBERTure import TemBERTure
n=0
models = {}
for p in range(len(categories_path)):
    for i in range(1, 4): #n replicas
        path=f"{main_path}{categories_path[p]}/3_REPLICAS/*replica{i}/output/checkpoint*_best_model/"
        path = glob.glob(path)[0]
        print(path)
        #path='/ibmm_data/TemBERTure/model/BERT_regr/2_LAYER_HEAD/3_REPLICAS/RANDOM_IND_lr1e-3_headdrop0.3_replica3/output/checkpoint-26320_epoch14_best_model/'
        model = TemBERTure(adapter_path=path, device='cuda')
        models[f'Model {categories[p]} - Replica {i}'] = model
        
        n += 1
print(models)
print('# models',len(models))


#######################################
############ 2. Data set ##############
#######################################
import pandas as pd
test_set = pd.read_csv(data_path,header=None)
test_set.columns = ['id','sequence','cls_label','tm','id2','species']
test_set

#######################################
############### 3. Data ###############
#######################################

# List of 18 models 
models = models

# data set 
X_val, y_val = test_set['sequence'],test_set['tm']

#######################################
############ 4. Ensemble: #############
#######################################

###################### METHOD 1: Greedy Algorithm Ensemble


from analysis.ensemble import GreedyAlgorithmEnsemble
print('*** GreedyAlgorithmEnsemble  ***')
selected_models, best_mae = GreedyAlgorithmEnsemble(models=models,X_val=X_val, y_val=y_val).ensemble()

# Output the final selected models and their performance
print("Final selected models:", selected_models)
print("Best MAE:", best_mae)
print('')
print('')

###################### METHOD 2: Weighted Ensemble (with budget)

from analysis.ensemble import WeightedAverageEnsemble
print('*** WeightedAverageEnsemble ***')
selected_models, best_mae = WeightedAverageEnsemble(models=models,X_val=X_val, y_val=y_val,budget=9).ensemble()

# Output the final selected models and their performance
print("Final selected models:", selected_models)
print("Best MAE:", best_mae)
print('')
print('')

###################### METHOD 3: Classification Based Ensemble

from temBERTure.temBERTure import TemBERTure
classifier = TemBERTure(adapter_path='/ibmm_data/TemBERTure/model/BERT_cls/BEST_MODEL/lr_1e-5_headropout01/output/best_model_epoch4/', task ='classification')
#classifier.model

from analysis.ensemble import ClassificationBasedEnsemble
print('*** ClassificationBasedEnsemble ***')
models_non_thermo, models_thermo = ClassificationBasedEnsemble(regression_models=models,seq=X_val, Tm=y_val,classifier_model=classifier).ensemble()
print('')
print('')



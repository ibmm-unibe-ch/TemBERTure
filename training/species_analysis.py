import os
import pandas as pd
from evaluate import classification_test_performances_withpreds, regression_test_performances



def species_analysis(df_path,wd,task):
    
    from utils import formatter
    import logging
    test_logger = logging.getLogger("Logger1")
    logger = logging.FileHandler('species_test.log')
    logger.setFormatter(formatter)
    test_logger.addHandler(logger)
    test_logger.setLevel(logging.INFO)

    
    print('#########################################################')
    test_logger.info(f'RESULTS FROM: {df_path}')
    test_logger.info(f'TASK: {task}')
    test_logger.info('')

    # Carica il file CSV in un DataFrame
    df = pd.read_csv(df_path,header=0)

    # Raggruppa il DataFrame per la colonna 'species'
    grouped = df.groupby('species')
    print(df['species'].unique())
    for species, group_data in grouped:
        # Imposta una nuova cartella di lavoro per ciascun gruppo di 'species'
        species_folder = os.path.join(wd, species)
        os.makedirs(species_folder, exist_ok=True)
        os.chdir(species_folder)
        test_logger.info(f"----- ANALYSIS FOR THE GROUP: {species} -----")
        
        preds = group_data['prediction']
        
        if task == 'regression':
            true_value = group_data['tm']
            from evaluate import regression_test_performances
            regression_test_performances(true_value, preds)
        elif task == 'classification' or task == 'classification_on_regression_data':
            labels = group_data['cls_label']
            from evaluate import classification_test_performances_withpreds
            classification_test_performances_withpreds(preds, labels)
        
        # Torna alla cartella di lavoro iniziale
        os.chdir(wd)
        test_logger.info('')
        test_logger.info('')

    return True

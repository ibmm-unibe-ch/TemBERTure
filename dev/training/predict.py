
def classification_preds(logits):
    import numpy as np
    logits = np.array(logits)
    preds = 1 / (1 + np.exp(-logits))
    predicted_labels = (preds > 0.5).astype(int)# Trasforma le probabilità in etichette binarie
    labels = np.array(labels).astype(int)
    return labels, predicted_labels



def predict(task, df,seq_col,model_name,model_type,adapters=None,adapter_path=None):
    import pandas as pd
    df_name = str(df)
    parti = df_name.split("/")
    df_name = parti[-2] + "_" + parti[-1]
    
    df = pd.read_csv(df, header=None, sep=',')
    
    print('**** RUNNING INFERENCE ****')
    print(f'Inference on data: {df} on column number {seq_col}')
    print(' !!! DATA MUST BE IN COMMA SEPARATED FORMAT WITH NO HEADER !!! ')
    print(f'Model: {model_type}')
    print(f'With adapters set to {adapters}')
    
    if adapters:
        print(f'Using adapters as best model from {adapter_path}')
    
    if "BERT" in model_type.upper():
        df[seq_col] = [" ".join("".join(sample.split())) for sample in df[seq_col]] #as required by rostlab models by their tokeizer
        model_type = 'BERTSequential'
        
    from model import model_init
    tokenizer, model = model_init(model_type, model_name, adapters,adapter_path)
    
    import math
    BATCH_SIZE = 16
    nb_batches = math.ceil(len(df)/BATCH_SIZE)
    y_preds = []
    
    import tqdm
    for i in tqdm.tqdm(range(nb_batches)):
        input_texts = df[i * BATCH_SIZE: (i+1) * BATCH_SIZE][seq_col].values.tolist()
        encoded = tokenizer(input_texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to('cuda')
        y_preds += model(**encoded).logits.reshape(-1).tolist()
        
    if task == 'classification':
        y_preds, y_preds_score = classification_preds(y_preds)
        df[len(df.columns)] = y_preds
        df[len(df.columns)] =  y_preds_score   
    else:
        df[len(df.columns)] = y_preds
            
    df[seq_col] = ["".join("".join(sample.split())) for sample in df[seq_col]]
    
    df_out_name = df_name + '_preds_out.txt'
    print(f'Saving the predicted out in {df_out_name}')
    df.to_csv(df_out_name,header=False,index=False)
    
    return print('Inference ended')
            


            
        
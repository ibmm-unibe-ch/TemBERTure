import matplotlib.pyplot as plt
import math
import pandas as pd
from datasets import Dataset
import torch
import tqdm
import logging 
import seaborn as sns

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
from utils import formatter
import logging
test_logger = logging.getLogger("Logger1")
logger = logging.FileHandler('test.log')
logger.setFormatter(formatter)
test_logger.addHandler(logger)
test_logger.setLevel(logging.INFO)


def regression_test_data(raw_test_ds):
    # TEST SET DATA PREP
    raw_test_ds = pd.read_csv(raw_test_ds, header=None)
    raw_test_ds.columns =['id','input_text','cls_label','target_text','id2','species']
    test_logger.info(f'# TEST SET {len(raw_test_ds)}')
    test_logger.info(f"# TEST SET 3 COLUMNS PREV {raw_test_ds[['id','input_text','target_text']].head(3)}")
    raw_test_ds["input_text"] = [" ".join("".join(sample.split())) for sample in raw_test_ds['input_text']]
    
    return raw_test_ds

def classifier_test_data(raw_test_ds):
    # TEST SET DATA PREP
    raw_test_ds = pd.read_csv(raw_test_ds, header=None)
    raw_test_ds.columns =['id','input_text','target_text','tm','id2','species'] 
    test_logger.info(f'# TEST SET {len(raw_test_ds)}')
    test_logger.info(f"# TEST SET 3 COLUMNS PREV {raw_test_ds[['id','input_text','target_text']].head(3)}")
    raw_test_ds['input_text'] = [" ".join("".join(sample.split())) for sample in raw_test_ds['input_text']]
    
    return raw_test_ds



def regression_test_performances(true_value, preds):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    results_dict = {
        "Pearson Correlation": pearsonr(preds, true_value)[0],
        "Spearman Correlation": spearmanr(preds, true_value)[0],
        "Mean Squared Error": mean_squared_error(true_value, preds),
        "Mean Absolute Error": mean_absolute_error(true_value, preds),
        "R-squared (R^2)": r2_score(true_value, preds)
    }
    
    test_logger.info(f"Pearson Correlation: {results_dict['Pearson Correlation']}")
    test_logger.info(f"Spearman Correlation: {results_dict['Spearman Correlation']}")
    test_logger.info(f"Mean Squared Error: {results_dict['Mean Squared Error']}")
    test_logger.info(f"Mean Absolute Error: {results_dict['Mean Absolute Error']}")
    test_logger.info(f"R-squared (R^2): {results_dict['R-squared (R^2)']}")
    
    # pred vs true plot
    plt.figure(figsize=(10,10))
    plt.scatter(true_value, preds, c='crimson',s=5)

    p1 = max(max(preds), max(true_value))
    p2 = min(min(preds), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Tm', fontsize=15)
    plt.ylabel('Tm predicted', fontsize=15)
    plt.axis('equal')
    plt.savefig('true_vs_pred_tm_.png')
    
    return preds, true_value



def classification_test_performances(logits, labels):
    from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, matthews_corrcoef
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    test_logger.info('*** EVALUATION ON TEST DATA  ***')

    logits = np.array(logits)
    preds = 1 / (1 + np.exp(-logits))  # Applica la funzione sigmoide ai logits
    
    predicted_labels = (preds > 0.5).astype(int)# Trasforma le probabilità in etichette binarie
    labels = np.array(labels).astype(int)
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    mcc = matthews_corrcoef(labels, predicted_labels)

    
    fpr, tpr, thresholds = roc_curve(labels, predicted_labels)
    lr_precision, lr_recall, _ = precision_recall_curve(labels, predicted_labels)

    
    roc_auc = auc(fpr, tpr)
    test_logger.info(f'AUC: {roc_auc:.4f}')

    # computing MC Coefficients
    MC = matthews_corrcoef(labels, predicted_labels)
    test_logger.info(f'Matthews Correlation Coefficient computed after applying the tuned/selected threshold : {MC}')


    # Get accuracy over the test set
    accuracy = accuracy_score(labels, predicted_labels)
    test_logger.info(f'Accuracy on test set: {accuracy*100:.2f}%')

    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic Test Data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC_testdata_.png')
    plt.clf()


    # plot the precision-recall curves
    plt.title('Precision-Recall curves on Test Data')
    no_skill = len(labels[labels==1]) / len(labels)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random Model')
    plt.plot(lr_recall, lr_precision, marker='.', label='Classifier')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig('precision_recall_test_.png')
    plt.clf()

    # Creating classification report
    test_logger.info('Classification report for TEST DATA:')
    test_logger.info(classification_report(labels,predicted_labels))
    unique_label = np.unique([labels, predicted_labels])
    cm = pd.DataFrame(
    confusion_matrix(labels, predicted_labels, labels=unique_label), 
    index=['true:{:}'.format(x) for x in unique_label], 
    columns=['pred:{:}'.format(x) for x in unique_label]
    )
    test_logger.info(cm)

    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=['Meso','Thermo'], yticklabels=['Meso','Thermo'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix on Test Data respect to Actual Category')
    plt.show(block=False)
    plt.savefig('cm_testdata.png')
    
    test_logger.info(f' "Accuracy": {accuracy}, "Precision": {precision}, "Recall": {recall}, "F1": {f1}, "MCC": {mcc}')

    return preds, predicted_labels, labels
    
def classification_test_performances_withpreds(preds, labels):
    
    from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, matthews_corrcoef
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    test_logger.info('*** EVALUATION ON TEST DATA  ***')
    preds = np.array(preds)
    predicted_labels = (preds > 0.5).astype(int)# Trasforma le probabilità in etichette binarie
    labels = np.array(labels).astype(int)
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    mcc = matthews_corrcoef(labels, predicted_labels)

    
    fpr, tpr, thresholds = roc_curve(labels, predicted_labels)
    lr_precision, lr_recall, _ = precision_recall_curve(labels, predicted_labels)

    
    roc_auc = auc(fpr, tpr)
    test_logger.info(f'AUC: {roc_auc:.4f}')
    print((f'AUC: {roc_auc:.4f}'))

    # computing MC Coefficients
    MC = matthews_corrcoef(labels, predicted_labels)
    test_logger.info(f'Matthews Correlation Coefficient computed after applying the tuned/selected threshold : {MC}')
    print(f'Matthews Correlation Coefficient computed after applying the tuned/selected threshold : {MC}')

    # Get accuracy over the test set
    accuracy = accuracy_score(labels, predicted_labels)
    test_logger.info(f'Accuracy on test set: {accuracy*100:.2f}%')
    print(f'Accuracy on test set: {accuracy*100:.2f}%')

    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic Test Data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC_testdata_.png')
    plt.clf()


    # plot the precision-recall curves
    plt.title('Precision-Recall curves on Test Data')
    no_skill = len(labels[labels==1]) / len(labels)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random Model')
    plt.plot(lr_recall, lr_precision, marker='.', label='Classifier')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig('precision_recall_test_.png')
    plt.clf()

    # Creating classification report
    test_logger.info('Classification report for TEST DATA:')
    test_logger.info(classification_report(labels,predicted_labels))
    print('Classification report for TEST DATA:')
    print(classification_report(labels,predicted_labels))
    
    unique_label = np.unique([labels, predicted_labels])
    cm = pd.DataFrame(
    confusion_matrix(labels, predicted_labels, labels=unique_label), 
    index=['true:{:}'.format(x) for x in unique_label], 
    columns=['pred:{:}'.format(x) for x in unique_label]
    )
    test_logger.info(cm)
    print(cm)

    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=['Meso','Thermo'], yticklabels=['Meso','Thermo'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix on Test Data respect to Actual Category')
    plt.show(block=False)
    plt.savefig('cm_testdata.png')
    
    test_logger.info(f' "Accuracy": {accuracy}, "Precision": {precision}, "Recall": {recall}, "F1": {f1}, "MCC": {mcc}')
    print(f' "Accuracy": {accuracy}, "Precision": {precision}, "Recall": {recall}, "F1": {f1}, "MCC": {mcc}')

    return preds, predicted_labels, labels
    


def evaluate_out(task,model,tokenizer,raw_test_df,best_model_path,BATCH_SIZE):

    
    test_logger.info(f'TEST DATA:{raw_test_df}')
    
    # TASK 
    if task == 'regression':
        raw_test_df = regression_test_data(raw_test_df)
    elif task =='classification':
        raw_test_df = classifier_test_data(raw_test_df)
    elif task =='classification_on_regression_data':
        raw_test_df = regression_test_data(raw_test_df)
        raw_test_df.columns =['id','input_text','target_text','tm','id2','species']
    elif task =='regression_on_classification_data':
        raw_test_df = classifier_test_data(raw_test_df)
        raw_test_df.columns =['id','input_text','cls_label','target_text','id2','species'] 
    elif task=='bacdive_sequence_classification':
        raw_test_df = pd.read_csv(raw_test_df, header=None)
        raw_test_df.columns =['id','input_text','target_text']
        test_logger.info(f'# TEST SET {len(raw_test_df)}')
        test_logger.info(f"# TEST SET 3 COLUMNS PREV {raw_test_df[['id','input_text','target_text']].head(3)}")
        raw_test_df["input_text"] = [" ".join("".join(sample.split())) for sample in raw_test_df['input_text']]

        
    test_logger.info(f' ** TEST FOR TASK {task} \n TEST DATA: {raw_test_df} \n USING BEST MODEL: {best_model_path}')
    
    nb_batches = math.ceil(len(raw_test_df)/BATCH_SIZE)
    y_preds = []

    for i in tqdm.tqdm(range(nb_batches)):
        input_texts = raw_test_df[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["input_text"].values.tolist()
        encoded = tokenizer(input_texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to('cuda')
        y_preds += model(**encoded).logits.reshape(-1).tolist()
    
    input_labels = raw_test_df["target_text"].values.tolist()
    df = pd.DataFrame(raw_test_df)

    # PERFORMANCES REPORT
    if task == 'regression' or task == 'regression_on_classification_data':
        preds, true_value = regression_test_performances(input_labels, y_preds)
        df[7] = preds
        df.columns =['id','sequence','cls_label','tm','id2','species','prediction']
    elif task =='classification' or task =='classification_on_regression_data':
        preds, predicted_labels, labels = classification_test_performances(y_preds, input_labels)
        df[8] = preds
        df[7] = predicted_labels
        df.columns =['id','sequence','cls_label','tm','id2','species','prediction_score','prediction']
    elif task == 'bacdive_sequence_classification': #non_redundunt_test_sequences_data_filtered_filtered_finaltestset
        preds, predicted_labels, labels = classification_test_performances(y_preds, input_labels)
        df[4] = preds
        df[3] = predicted_labels
        df.columns =['id','sequence','cls_label','prediction','prediction_score']
    
    df["sequence"] = ["".join("".join(sample.split())) for sample in df['sequence']]
    df.to_csv('test_out.txt',header=True,index=False)

    
    



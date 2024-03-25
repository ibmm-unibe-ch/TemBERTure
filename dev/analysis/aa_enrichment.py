import argparse
parser = argparse.ArgumentParser(description='')

# data
parser.add_argument("--model_path", help="", type=str,default=False)
parser.add_argument("--test_out", help="", type=str,default=False)
parser.add_argument("--rostlab", help="", type=bool,default=False)
args = parser.parse_args()
model_path_adapter = args.model_path  #'/ibmm_data/TemBERTure/model/BERT_cls/BEST_MODEL/lr_1e-5_headropout01/output/best_model_epoch4/'
test_out = args.test_out #'/ibmm_data/TemBERTure/model/BERT_cls/BEST_MODEL/lr_1e-5_headropout01/TEST/test_classifier_data/test_out.txt'

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

##  AA ENRICHMENT ANALYSIS 
from attention_utils import *

"""# TemBERTure INITIALITATION AND ATTENTION SCORE READING FUNCTIONS AND OUTLIER IDENTIFICATION"""
# Uploading the TemBERTure bert model, fragments trained and initialize it for attention analysis
model_bert = Rostlab_for_attention() if args.rostlab else TemBERTure_for_attention(model_path_adapter)
## rostlab only (REMEMBER TO CHANGE IT MANUALLY):
#model_bert = Rostlab_for_attention()
#print(model_bert)


"""# DATA """
# model performance
test_out=pd.read_csv(test_out, header=0)
test_out['SEQ_LENGTH']=test_out['sequence'].str.len()
print(test_out['cls_label'])

print('Max seq length:',max(test_out['SEQ_LENGTH']))
print('Min seq length:',min(test_out['SEQ_LENGTH']))

meso_seqs=test_out[test_out['cls_label']==0]
thermo_seqs=test_out[test_out['cls_label']==1]

print('# SEQUENCES in the test.out:',len(test_out['id']))
print('# MESO SEQUENCES in the test.out:',len(meso_seqs['id'])) 
print('# THERMO SEQUENCES in the test.out:',len(thermo_seqs['id']))


########## ONLY CORRECT PREDICTED SEQUENCES ######################
correct_seqs = test_out[((test_out['cls_label'] == 0) & (test_out['prediction_score'] <= 0.5)) | ((test_out['cls_label'] == 1) & (test_out['prediction_score'] > 0.5)) ]
meso_seqs=correct_seqs[correct_seqs['cls_label']==0]
thermo_seqs=correct_seqs[correct_seqs['cls_label']==1]
print('# SEQUENCES in the correct_seqs:',len(correct_seqs['id']))
print('# MESO SEQUENCES in correct_seqs:',len(meso_seqs['id'])) 
print('# THERMO SEQUENCES in the correct_seqs:',len(thermo_seqs['id'])) 


"""## AA ENRICHMENT - FUNCTION"""


def aa_enrichment_seqs(data,info_seq_out):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    import tqdm 
    import json
    import prody
    import copy

    ##################################################################################################
    ######################## INFO STORAGE ############################################################
    ##################################################################################################

    #EACH SEQUENCE % AA COMPOSITION FOR ALL SEQS
    aa_comp={'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],}
    #COUNTER AA IN EACH SEQ, FOR ALL SEQS
    aa_distr_count={'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],}

    # AAs (COUNTER) ASSOCIATED WITH HIGH ATTENTION SCORE
    high_att_score_aa={'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],}

    f = open(info_seq_out, mode="a")
    seq_info=[]

    ##################################################################################################
    ######################## FOR EACH SEQ ############################################################
    ##################################################################################################
    s = 0
    for seq in tqdm.tqdm(data['sequence']):
        device = 'cpu' if len(seq) > 2500 else 'cuda'
 
        seq_list=seq
        print(f'*** # SEQ {s}')
        print(f'*** LENGTH {len(seq)}')
        
        if len(seq) > 17000:
            print('**** sequence too long, skipped')
            continue 
        
        #if len(seq) < 512:
        #   print('Aminoacid sequence length < 510')
        #else:
        #    print('Aminoacid sequence length > 510')
        #print('check if passed successfully')

        #2. aminoacid composition in % per seq
        X = ProteinAnalysis(seq)
        aa_comp['L'].append(X.get_amino_acids_percent()['L'])
        aa_comp['A'].append(X.get_amino_acids_percent()['A'])
        aa_comp['G'].append(X.get_amino_acids_percent()['G'])
        aa_comp['V'].append(X.get_amino_acids_percent()['V'])
        aa_comp['E'].append(X.get_amino_acids_percent()['E'])
        aa_comp['S'].append(X.get_amino_acids_percent()['S'])
        aa_comp['I'].append(X.get_amino_acids_percent()['I'])
        aa_comp['K'].append(X.get_amino_acids_percent()['K'])
        aa_comp['R'].append(X.get_amino_acids_percent()['R'])
        aa_comp['D'].append(X.get_amino_acids_percent()['D'])
        aa_comp['T'].append(X.get_amino_acids_percent()['T'])
        aa_comp['P'].append(X.get_amino_acids_percent()['P'])
        aa_comp['N'].append(X.get_amino_acids_percent()['N'])
        aa_comp['Q'].append(X.get_amino_acids_percent()['Q'])
        aa_comp['F'].append(X.get_amino_acids_percent()['F'])
        aa_comp['Y'].append(X.get_amino_acids_percent()['Y'])
        aa_comp['M'].append(X.get_amino_acids_percent()['M'])
        aa_comp['H'].append(X.get_amino_acids_percent()['H'])
        aa_comp['C'].append(X.get_amino_acids_percent()['C'])
        aa_comp['W'].append(X.get_amino_acids_percent()['W'])

        # 2. aminoacid composition - FREQUENCY (counter)
        # it is different from the previous step since here we are counting the aminoacid
        # aa_composition_count in step 3.1

        # 3
        for i in range(0,len(seq_list)): 
            aa_distr_count[seq_list[i]].append(1)

        # 4. computing attention respect to cls token
        df_all_vs_all,att_to_cls,df_att_to_cls_exp = attention_score_to_cls_token_and_to_all(seq.replace('',' '),model_bert.to(device),device)
        att_to_cls = att_to_cls.drop(labels = ['[CLS]','[SEP]'])

        # 5. high attention analysis
        df = pd.DataFrame({'att_score': att_to_cls}) #attention without any standardization
        attention_outliers_df=find_high_outliers_IQR(df) #outlier = high attention 

        for a in list(attention_outliers_df.index.values): # saving in a counter which aa ('L','H' etc) has high attention
            high_att_score_aa[a].append(1)

        #5. store all the seq info
        seq_info.append({'aa': list(df.index.values.tolist()),'att_score':list(att_to_cls.tolist())})
        s += 1


    ##################################################################################################
    ######################## OVERALL #################################################################
    ##################################################################################################

    # 1. saving all the seq info (aas,attention score) in a json
    print(f'Saving for each sequence the aa-att_score in a json file {info_seq_out}')
    json.dump(seq_info, f)
    f.close()

    # 2. AVG AMINOACID COMPOSITION IN % (on all the seqs)
    # ex: on average the x% of protein sequences is composed by L
    print('Computing the average AAs composition ...')
    aa_comp_avgDict = {}
    for k,v in tqdm.tqdm(aa_comp.items()):
        aa_comp_avgDict[k] = sum(v)/ float(len(v))
        if float(len(v)) != len(data):
            print('Error')
            print('Tot sequence analyzed',len(data))
    #sum(aa_composition_avgDict.values()) # should be almost equal to 1
    
    # 2.1 AVG AMINOACID DISTRIBUTION (FREQUENCY) IN % (on all the seqs)
    # ex: the x% of all the aas are L
    print('Computing the average AAs distribution ...')
    aa_distr_count_avgDict = {}
    for k,v in (aa_distr_count.items()):
        # v is the list of grades for student k
        aa_distr_count_avgDict[k] = sum(v)/ float(sum([sum(aas)  for aas in aa_distr_count.values()]))


    # 3. HIGH ATT SCORE IN % - FOR EACH AA
    # for example the x% of all high attention scores focus on 'L;
    high_att_score_aa_avgDict = {k: (sum(high_att_score_aa[k])) for k in high_att_score_aa.keys()}
    high_att_score_aa_avgDict = {k: (high_att_score_aa_avgDict[k]/sum(high_att_score_aa_avgDict.values())) for k in high_att_score_aa_avgDict.keys()}

    return aa_comp,aa_comp_avgDict,aa_distr_count,aa_distr_count_avgDict,high_att_score_aa,high_att_score_aa_avgDict

"""# 1.ANALYSIS - AA ENRICHMENT"""

""" ## 1.1 Computation """


"""### MESO"""

meso_aa_comp,meso_aa_comp_avgDict,meso_aa_distr_count,meso_aa_distr_count_avgDict,meso_high_att_score_aa,meso_high_att_score_aa_avgDict = aa_enrichment_seqs(meso_seqs,'meso_seqs_info.json')

print('# aas in all MESO sequences:')
meso_n_all_aa=sum([sum(aas)  for aas in meso_aa_distr_count.values()])
print(meso_n_all_aa)


import numpy as np
np.save('meso_high_att_score_aa_avgDict.npy',meso_high_att_score_aa_avgDict,allow_pickle=True)
np.save('meso_aa_distr_count_avgDict.npy',meso_aa_distr_count_avgDict,allow_pickle=True)
np.save('meso_aa_distr_count.npy',meso_aa_distr_count,allow_pickle=True)
np.save('meso_high_att_score_aa.npy',meso_high_att_score_aa,allow_pickle=True)


'''### THERMO'''


thermo_aa_comp,thermo_aa_comp_avgDict,thermo_aa_distr_count,thermo_aa_distr_count_avgDict,thermo_high_att_score_aa,thermo_high_att_score_aa_avgDict = aa_enrichment_seqs(thermo_seqs,'thermo_seqs_info.json')

print('# aas in all THERMO sequences:')
thermo_n_all_aa=sum([sum(aas)  for aas in thermo_aa_distr_count.values()])
print(thermo_n_all_aa)

import numpy as np
np.save('thermo_high_att_score_aa_avgDict.npy',thermo_high_att_score_aa_avgDict,allow_pickle=True)
np.save('thermo_aa_distr_count_avgDict.npy',thermo_aa_distr_count_avgDict,allow_pickle=True)
np.save('thermo_aa_distr_count.npy',thermo_aa_distr_count,allow_pickle=True)
np.save('thermo_high_att_score_aa.npy',thermo_high_att_score_aa,allow_pickle=True)


## _aa_distr_count.npy --> contains the frequency of each aa in all the subset of sequences (meso subset or thermo subset) (it contains also the one classified as HAS)
## _high_att_score_aa --> contains the frequency of each HAS aa in all the subset of sequences (meso subset or thermo subset)

import argparse
parser = argparse.ArgumentParser(description='')

# data
parser.add_argument("--model_path", help="", type=str,default=False)
parser.add_argument("--pdb_thermo", help="", type=str,default=False)
parser.add_argument("--pdb_meso", help="", type=str,default=False)
parser.add_argument("--rostlab", help="", type=bool,default=False)
args = parser.parse_args()
model_path_adapter = args.model_path  #'/ibmm_data/TemBERTure/model/BERT_cls/BEST_MODEL/lr_1e-5_headropout01/output/best_model_epoch4/'
pdb_thermo = args.pdb_thermo
pdb_meso = args.pdb_meso

##  AA ENRICHMENT ANALYSIS for PDB DATA
from attention_utils import *

"""# TemBERTure INITIALITATION AND ATTENTION SCORE READING FUNCTIONS AND OUTLIER IDENTIFICATION"""
# Uploading the TemBERTure bert model, fragments trained and initialize it for attention analysis
model_bert = Rostlab_for_attention() if args.rostlab else TemBERTure_for_attention(model_path_adapter)
print(model_bert)

"""# DATA """

#### ASPETTARE SYMELA PERCHE ALTRA VOLTA MI AVEVA DATO MASK PERCHE LE SEQUENZE DIFFERISCONO NEI PDB
#### We run the model on complete_seq and plot in the .pdb only the amino acid with mask 1
#### remember to *** USING ONLY SEQUENCE PREDICTED CORRECTLY ***

"""# FUNCTION """
def aa_enrichment_with_sec_str(data,info_seq_out):
  '''Function to compute aminoacid propensity vs high attention score enrichment extrappolating at the same time secondary structure information.
  Data must be pdb id,not just sequences'''
  '''AA ENRICHMENT WITH SECONDARY STRUCTURE EXTRACTION - ONLY PDB ID DATA'''
  from Bio.SeqUtils.ProtParam import ProteinAnalysis
  import tqdm 
  import json
  import prody
  import copy
  import os

  data = pd.read_csv(data,sep=',',header=None)
  print(data)
  
  ##################################################################################################
  ######################## INFO STORAGE ############################################################
  ##################################################################################################

  #EACH SEQUENCE % AA COMPOSITION FOR ALL SEQS
  aa_composition={'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],}
  #EACH SEQUENCE COUNTER AA COMPOSITION FOR ALL SEQS
  aa_composition_count={'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],}

  # SECONDARY STRUCTURE COMPOSITION (COUNTER) FOR ALL THE SEQS + DETAIL OF WHICH AA
  secondary_structure_composition={'H':{'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],},
  'E':{'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],},
  'G':{'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],},
  'I':{'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],},
  'C':{'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],}}

  # SECONDARY STRUCTURE COMPOSITION (COUNTER) FOR EACH AA 
  aa_vs_sec_str_composition={'L':{'H':[],'E':[],'G':[],'I':[],'C':[]},'A':{'H':[],'E':[],'G':[],'I':[],'C':[]},'G':{'H':[],'E':[],'G':[],'I':[],'C':[]},'V':{'H':[],'E':[],'G':[],'I':[],'C':[]},'E':{'H':[],'E':[],'G':[],'I':[],'C':[]},'S':{'H':[],'E':[],'G':[],'I':[],'C':[]},'I':{'H':[],'E':[],'G':[],'I':[],'C':[]},'K':{'H':[],'E':[],'G':[],'I':[],'C':[]},'R':{'H':[],'E':[],'G':[],'I':[],'C':[]},'D':{'H':[],'E':[],'G':[],'I':[],'C':[]},'T':{'H':[],'E':[],'G':[],'I':[],'C':[]},'P':{'H':[],'E':[],'G':[],'I':[],'C':[]},'N':{'H':[],'E':[],'G':[],'I':[],'C':[]},'Q':{'H':[],'E':[],'G':[],'I':[],'C':[]},'F':{'H':[],'E':[],'G':[],'I':[],'C':[]},'Y':{'H':[],'E':[],'G':[],'I':[],'C':[]},'M':{'H':[],'E':[],'G':[],'I':[],'C':[]},'H':{'H':[],'E':[],'G':[],'I':[],'C':[]},'C':{'H':[],'E':[],'G':[],'I':[],'C':[]},'W':{'H':[],'E':[],'G':[],'I':[],'C':[]},}

  # AAs (COUNTER) ASSOCIATED WITH HIGH ATTENTION SCORE
  high_att_score_aminoacid={'L':[],'A':[],'G':[],'V':[],'E':[],'S':[],'I':[],'K':[],'R':[],'D':[],'T':[],'P':[],'N':[],'Q':[],'F':[],'Y':[],'M':[],'H':[],'C':[],'W':[],}

  # AAs (COUNTER) ASSOCIATED WITH HIGH ATTENTION SCORE  + SEC STR FOR EACH HIGH ATTENTION AA 
  high_att_score_aminoacid_with_sec_str={'L':{'H':[],'E':[],'G':[],'I':[],'C':[]},'A':{'H':[],'E':[],'G':[],'I':[],'C':[]},'G':{'H':[],'E':[],'G':[],'I':[],'C':[]},'V':{'H':[],'E':[],'G':[],'I':[],'C':[]},'E':{'H':[],'E':[],'G':[],'I':[],'C':[]},'S':{'H':[],'E':[],'G':[],'I':[],'C':[]},'I':{'H':[],'E':[],'G':[],'I':[],'C':[]},'K':{'H':[],'E':[],'G':[],'I':[],'C':[]},'R':{'H':[],'E':[],'G':[],'I':[],'C':[]},'D':{'H':[],'E':[],'G':[],'I':[],'C':[]},'T':{'H':[],'E':[],'G':[],'I':[],'C':[]},'P':{'H':[],'E':[],'G':[],'I':[],'C':[]},'N':{'H':[],'E':[],'G':[],'I':[],'C':[]},'Q':{'H':[],'E':[],'G':[],'I':[],'C':[]},'F':{'H':[],'E':[],'G':[],'I':[],'C':[]},'Y':{'H':[],'E':[],'G':[],'I':[],'C':[]},'M':{'H':[],'E':[],'G':[],'I':[],'C':[]},'H':{'H':[],'E':[],'G':[],'I':[],'C':[]},'C':{'H':[],'E':[],'G':[],'I':[],'C':[]},'W':{'H':[],'E':[],'G':[],'I':[],'C':[]},}

  # HIGH ATTENTION SCORE (COUNTER) SEC STR DISTRIBUTION
  high_att_score_sec_str={'H':[],'E':[],'G':[],'I':[],'C':[]}

  f = open(info_seq_out, mode="a")
  seq_info=[]

  n_pdb_processed = 0
  ##################################################################################################
  ######################## FOR EACH SEQ ############################################################
  ##################################################################################################

  for coord in tqdm.tqdm(data[0]): #pdb id
    print(coord)

    # 1. reading pdb
    pdb_id,pdb_chain=coord.rsplit('_') 

    pdb = prody.parsePDB(pdb_id,chain=pdb_chain)
    print('')
    print(f'Reading {pdb_id} with chain {pdb_chain}')
    print('')

    seq=pdb.ca.getSequence()
    seq_list=seq
    device = 'cpu' if len(seq) > 2500 else 'cuda'
    if len(seq) < 512:
      print('Aminoacid sequence length < 510')
    else:
      print('Aminoacid sequence length > 510')
      pass

    print('check if passed successfully')

    #2. aminoacid composition in % 
    X = ProteinAnalysis(seq)
    aa_composition['L'].append(X.get_amino_acids_percent()['L'])
    aa_composition['A'].append(X.get_amino_acids_percent()['A'])
    aa_composition['G'].append(X.get_amino_acids_percent()['G'])
    aa_composition['V'].append(X.get_amino_acids_percent()['V'])
    aa_composition['E'].append(X.get_amino_acids_percent()['E'])
    aa_composition['S'].append(X.get_amino_acids_percent()['S'])
    aa_composition['I'].append(X.get_amino_acids_percent()['I'])
    aa_composition['K'].append(X.get_amino_acids_percent()['K'])
    aa_composition['R'].append(X.get_amino_acids_percent()['R'])
    aa_composition['D'].append(X.get_amino_acids_percent()['D'])
    aa_composition['T'].append(X.get_amino_acids_percent()['T'])
    aa_composition['P'].append(X.get_amino_acids_percent()['P'])
    aa_composition['N'].append(X.get_amino_acids_percent()['N'])
    aa_composition['Q'].append(X.get_amino_acids_percent()['Q'])
    aa_composition['F'].append(X.get_amino_acids_percent()['F'])
    aa_composition['Y'].append(X.get_amino_acids_percent()['Y'])
    aa_composition['M'].append(X.get_amino_acids_percent()['M'])
    aa_composition['H'].append(X.get_amino_acids_percent()['H'])
    aa_composition['C'].append(X.get_amino_acids_percent()['C'])
    aa_composition['W'].append(X.get_amino_acids_percent()['W'])

    # 2. aminoacid composition - FREQUENCY (counter)
    # it is different from the previous step since here we are counting the aminoacid
    # aa_composition_count in step 3.1

    # 3. secondary structure by prody
    struct,pdb=prody.parsePDB(pdb_id, chain=pdb_chain, subset='calpha', header=True,secondary=True)
    sec_str_seq=list(struct.getSecstrs())

    # 3.1 secondary structure composition
    for i in range(0,len(seq_list)): 
      try:
        secondary_structure_composition[sec_str_seq[i]][seq_list[i]].append(1)
      except KeyError:
        pass
      try:
        aa_vs_sec_str_composition[seq_list[i]][sec_str_seq[i]].append(1)
      except KeyError:
        pass
      try:
        aa_composition_count[seq_list[i]].append(1)
      except KeyError:
        pass
        
    
    #sasa score (for each aa)
    #res_sasa,sasa_score=biopython_sasa_residue(pdb_id,pdb_chain)
    #print(len(sasa_score))
    #print(len(sec_str_seq))
    #if len(sasa_score) != len(sec_str_seq):
     # print('ATTENTION ---- sasa score with different length')

    # 4. computing attention respect to cls token
    df_all_vs_all,att_to_cls,df_att_to_cls_exp = attention_score_to_cls_token_and_to_all(seq.replace('',' '),model_bert.to(device),device)
    att_to_cls = att_to_cls.drop(labels = ['[CLS]','[SEP]'])

    # 5. high attention analysis
    df = pd.DataFrame({'att_score': att_to_cls,'sec_structure':sec_str_seq}) #attention without any standardization
    attention_outliers_df=find_high_outliers_IQR(df) #outlier = high attention 

    for a in list(attention_outliers_df.index.values): # saving in a counter which aa ('L','H' etc) has high attention
      high_att_score_aminoacid[a].append(1)

    for s in attention_outliers_df.sec_structure:  # saving in a counter which secondary structure is linked to high attention
      high_att_score_sec_str[s].append(1)

    for i in range(0,len(attention_outliers_df)): # saving in a counter which aa ('L','H' etc) has high attention plus the sec str distribution for each aa
      high_att_score_aminoacid_with_sec_str[attention_outliers_df.index.values[i]][attention_outliers_df.sec_structure[i]].append(1)
    
    #5. store all the seq info
    #seq_info.append({'aa': list(df.index.values.tolist()),'att_score':list(att_to_cls.tolist()),'sec_str':sec_str_seq,'sasa_score':sasa_score})
    seq_info.append({'aa': list(df.index.values.tolist()),'att_score':list(att_to_cls.tolist()),'sec_str':sec_str_seq})
    
    
    n_pdb_processed += 1
    [os.remove(file) for file in os.listdir() if file.endswith("pdb.gz")]

  ##################################################################################################
  ######################## OVERALL #################################################################
  ##################################################################################################

  # 1. saving all the seq info (aas,attention score,sasa score,secondary structure) in a json
  print(f'Saving for each sequence the aa-att_score-sec_str in a json file {info_seq_out}')
  json.dump(seq_info, f)
  f.close()

  # 2. AVG AMINOACID COMPOSITION IN % (on all the seqs)
  # ex: on average the x% of protein sequences is composed by L
  print('Computing the average AAs composition ...')
  aa_composition_avgDict = {}
  for k,v in tqdm.tqdm(aa_composition.items()):
      aa_composition_avgDict[k] = sum(v)/ float(len(v))
      if float(len(v)) != len(data):
        print('Error')
        print('Tot sequence analyzed',len(data))
  #sum(aa_composition_avgDict.values()) # should be almost equal to 1
   
  # 2.1 AVG AMINOACID DISTRIBUTION (FREQUENCY) IN % (on all the seqs)
  # ex:  we have x% L in protein sequences 
  print('Computing the average AAs composition ...')
  aa_composition_count_avgDict = {}
  for k,v in (aa_composition_count.items()):
      # v is the list of grades for student k
      aa_composition_count_avgDict[k] = sum(v)/ float(sum([sum(aas)  for aas in aa_composition_count.values()]))

  # 5. AVG SEC STRUCT COMPOSITION FOR EACH AA IN % (each aa sums to 1)
  aa_vs_sec_str_composition_count=copy.deepcopy(aa_vs_sec_str_composition)
  for k in aa_vs_sec_str_composition:
    tot_per_aa = sum([sum(aa_vs_sec_str_composition[k][sec]) for sec in aa_vs_sec_str_composition[k]])
    for sec in aa_vs_sec_str_composition[k]:
      aa_vs_sec_str_composition[k][sec]=[sum(aa_vs_sec_str_composition[k][sec])/(tot_per_aa)]

  # editing the dict to be plottable stacked
  # 10 % of L are H ---> 10% of the L are H * Percentage of L in the data = H in L comparable to all data
  aa_vs_sec_str_composition_plot=copy.deepcopy(aa_vs_sec_str_composition)
  for k in aa_vs_sec_str_composition_plot:
    for sec in aa_vs_sec_str_composition_plot[k]:
      aa_vs_sec_str_composition_plot[k][sec] = aa_composition_avgDict[k] * aa_vs_sec_str_composition_plot[k][sec][0]

  #same steps but in this case instead to plot it for the composition we work with the frequency counter
  
  aa_vs_sec_str_composition_count_plot=copy.deepcopy(aa_vs_sec_str_composition)
  for k in aa_vs_sec_str_composition_count_plot:
    for sec in aa_vs_sec_str_composition_count_plot[k]:
      aa_vs_sec_str_composition_count_plot[k][sec] = aa_composition_count_avgDict[k] * aa_vs_sec_str_composition_count_plot[k][sec][0]
  
  # 5. AVG SEC STRUCT COMPOSITION FOR ALL THE SEQS
  sec_struct_composition_avgDict_tmp = {}
  for k in secondary_structure_composition:
    sec_struct_composition_avgDict_tmp[k]=sum([item for sublist in secondary_structure_composition[k].values() for item in sublist])
  sec_struct_composition_avgDict = {}
  for k in sec_struct_composition_avgDict_tmp:
    sec_struct_composition_avgDict[k]=sec_struct_composition_avgDict_tmp[k]/sum(sec_struct_composition_avgDict_tmp.values()) 

  # 3. HIGH ATT SCORE IN % - FOR EACH AA
  # for example the x% of all high attention scores focus on 'L;
  high_att_score_aminoacid_avgDict = {k: (sum(high_att_score_aminoacid[k])) for k in high_att_score_aminoacid.keys()}
  high_att_score_aminoacid_avgDict = {k: (high_att_score_aminoacid_avgDict[k]/sum(high_att_score_aminoacid_avgDict.values())) for k in high_att_score_aminoacid_avgDict.keys()}
  
  # 4. HIGH ATT SCORE IN % - FOR EACH SEC STR
  # for example the x% of high attention scores focus on 'H;
  high_secondary_structure_avgDict = {k: (sum(high_att_score_sec_str[k])) for k in high_att_score_sec_str.keys()}
  high_secondary_structure_avgDict = {k: (high_secondary_structure_avgDict[k]/sum(high_secondary_structure_avgDict.values())) for k in high_secondary_structure_avgDict.keys()}

  # 6. HIGH ATT SCORE - AVG SEC STRUCT COMPOSITION FOR EACH AA IN % (each aa sums to 1)
  high_att_score_aminoacid_with_sec_str_count=copy.deepcopy(high_att_score_aminoacid_with_sec_str)
  for k in high_att_score_aminoacid_with_sec_str:
    tot_per_aa = sum([sum(high_att_score_aminoacid_with_sec_str[k][sec]) for sec in high_att_score_aminoacid_with_sec_str[k]])
    if tot_per_aa==0:
      continue
    for sec in high_att_score_aminoacid_with_sec_str[k]:
      high_att_score_aminoacid_with_sec_str[k][sec]=sum(high_att_score_aminoacid_with_sec_str[k][sec])/(tot_per_aa)
  # editing the dict to be plottable stacked
  high_att_score_aminoacid_with_sec_str_plot=copy.deepcopy(high_att_score_aminoacid_with_sec_str)
  # 10 % of high attention score classified as L have H as sec str ---> x % of the high att L are H * Percentage of L in the high attention score = H in L for the L classified as high attention score
  for k in high_att_score_aminoacid_with_sec_str_plot:
    for sec in high_att_score_aminoacid_with_sec_str_plot[k]:
      #print(high_att_score_aminoacid_with_sec_str_plot[k][sec])
      if not high_att_score_aminoacid_avgDict[k]:
        high_att_score_aminoacid_avgDict[k] = 0.0
      if not high_att_score_aminoacid_with_sec_str_plot[k][sec]:
        high_att_score_aminoacid_with_sec_str_plot[k][sec] = 0.0
      high_att_score_aminoacid_with_sec_str_plot[k][sec] = high_att_score_aminoacid_avgDict[k] * high_att_score_aminoacid_with_sec_str_plot[k][sec]

    

  return aa_composition,aa_composition_avgDict,aa_composition_count,aa_composition_count_avgDict,high_att_score_aminoacid,high_att_score_aminoacid_avgDict, n_pdb_processed

"""# 1.ANALYSIS - AA ENRICHMENT"""

""" ## 1.1 Computation """


"""### MESO"""

meso_aa_comp,meso_aa_comp_avgDict,meso_aa_distr_count,meso_aa_distr_count_avgDict,meso_high_att_score_aa,meso_high_att_score_aa_avgDict, n_pdb_processed = aa_enrichment_with_sec_str(pdb_meso,'pdb_meso_seqs_info.json')


print('Check number of meso pdb processed correctly:',n_pdb_processed)


import numpy as np
np.save('pdb_meso_high_att_score_aa_avgDict.npy',meso_high_att_score_aa_avgDict,allow_pickle=True)
np.save('pdb_meso_aa_distr_count_avgDict.npy',meso_aa_distr_count_avgDict,allow_pickle=True)
np.save('pdb_meso_aa_distr_count.npy',meso_aa_distr_count,allow_pickle=True)
np.save('pdb_meso_high_att_score_aa.npy',meso_high_att_score_aa,allow_pickle=True)



'''### THERMO'''

thermo_aa_comp,thermo_aa_comp_avgDict,thermo_aa_distr_count,thermo_aa_distr_count_avgDict,thermo_high_att_score_aa,thermo_high_att_score_aa_avgDict, n_pdb_processed = aa_enrichment_with_sec_str(pdb_thermo,'pdb_thermo_seqs_info.json')

print('Check number of thermo pdb processed correctly:',n_pdb_processed)


import numpy as np
np.save('pdb_thermo_high_att_score_aa_avgDict.npy',thermo_high_att_score_aa_avgDict,allow_pickle=True)
np.save('pdb_thermo_aa_distr_count_avgDict.npy',thermo_aa_distr_count_avgDict,allow_pickle=True)
np.save('pdb_thermo_aa_distr_count.npy',thermo_aa_distr_count,allow_pickle=True)
np.save('pdb_thermo_high_att_score_aa.npy',thermo_high_att_score_aa,allow_pickle=True)


## _aa_distr_count.npy --> contains the frequency of each aa in all the subset of sequences (meso subset or thermo subset) (it contains also the one classified as HAS)
## _high_att_score_aa --> contains the frequency of each HAS aa in all the subset of sequences (meso subset or thermo subset)

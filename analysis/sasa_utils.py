###### 
from .attention_utils import *

def upp3_to_1letter(x):
    if len(x) % 3 != 0:
        raise ValueError('Input length should be a multiple of three')

    y = ''
    for i in range(len(x) // 3):
            y += d[x[3 * i : 3 * i + 3]]
    return y

#################################################################
################### SASA & SALT BRIDGES ################### 
#################################################################

### SALT BRIDGES

'''The reason for selecting only oxygen and nitrogen atoms is that they are the atoms that can participate in salt bridges. In a salt bridge, a positively charged amino acid (e.g. Lysine, Arginine)
forms an electrostatic interaction with a negatively charged amino acid (e.g. Glutamic Acid, Aspartic Acid) through the interaction of their charged functional groups (NH3+ and COO-).
The positively charged amino acid contributes the NH3+ group, which contains a nitrogen atom, and the negatively charged amino acid contributes the COO- group, which contains an oxygen atom.

Regarding the acid_salt_bridges selection query, this is looking for C-alpha (name CA) atoms that are in the same residue as a Glutamic Acid or Aspartic Acid with an oxygen atom
(acid_sel) and within a distance of 4 Angstroms of a Histidine, Lysine, or Arginine with a nitrogen atom (basic_sel). This is a reasonable way to identify potential salt bridges in
a protein structure. Finally, the pdb.select method is used to apply the selection query to the PDB object and return a list of residue IDs (aa_id) that satisfy the query.'''

acid_sel="(protein and resname CYS and oxygen and not backbone)"
basic_sel="(protein and resname CYS and nitrogen and not backbone)"

acid_salt_bridges = f"name CA and same residue as ({acid_sel} and within 4 of {basic_sel})"

# salt bridge selection
acid_sel="(protein and resname GLU ASP and oxygen and not backbone)"
basic_sel="(protein and resname HIS LYS ARG and nitrogen and not backbone)"

acid_salt_bridges = f"name CA and same residue as ({acid_sel} and within 4 of {basic_sel})"
basic_salt_bridges = f"name CA and same residue as ({basic_sel} and within 4 of {acid_sel})"

def sasa_buried_not_buried_map(sasa_score):
    sasa=[10 if i<0.25 else 100 for i in sasa_score ]
    return sasa

def sasa_mask_selection(mask_col,sasa_score):
    boolean_col=mask_col.map({1: True, 0: False})
    new_sasa_col=[]
    n=1
    for i, x in enumerate(boolean_col):
        if x == True:
            n = n-1
            new_sasa_col.append(sasa_score[n])
            n +=2
        else:
            new_sasa_col.append('NaN')
    return new_sasa_col




def biopython_sasa_residue_and_salt_bridges(data,model_bert,device,sasa_on_pdb=False,att_on_pdb=False):
    ## check code everything correct 26/01/2024
    '''How to read biopython structure where a is the chain
    for resi_ind, resi in enumerate(struct[0]['A'].get_residues()):

    res_id = resi.get_full_id()
    print(res_id)
    print(res_id[3])
    resi_name=resi.get_resname()
    print(resi_name)'''
    
    ''' 
    1. SASA Score Thresholding:
        If sasa_on_pdb is set to True, the code calculates a map of buried and non-buried residues based on the SASA (Solvent Accessible Surface Area) score.
        It sets the B-factor (or beta-factor) of the PDB (Protein Data Bank) structure based on the calculated SASA values.
        Then, it writes the modified PDB structure with SASA values as beta factors to a PDB file.

    2. Attention Score Coloring:
        If att_on_pdb is True, the code computes an attention score for amino acids present in the PDB sequence.
        It sets the B-factor of the PDB structure based on the attention scores (attention score are first multiply by 10000).
        Subsequently, it writes the PDB structure with attention scores as beta factors to a PDB file.
        Additionally, it checks for consistency in the lengths of attention scores and the masked amino acid sequence.

The function sasa_buried_not_buried_map(sasa_score) is utilized to map SASA scores to buried or non-buried residues. It assigns a value of 10 if the SASA score is less than 0.25, otherwise assigns 100.
In essence, the code provides a method to visualize SASA scores and attention scores by utilizing the beta-factor field in PDB files, facilitating structural analysis and interpretation.'''
    #%shell
  # !gunzip *.gz

    from Bio.PDB import MMCIFParser,PDBParser
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.PDBList import PDBList
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.PDB.DSSP import DSSP
    import Bio.PDB.PDBExceptions
    import prody
    from collections import OrderedDict

    baseline_sasa={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}
    HAS_sasa={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}

    HAS_salt_bridge={'E':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'R':{'surface':[],'core':[]}}
    ALL_salt_bridge={'E':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'R':{'surface':[],'core':[]}}


    baseline_sasa=OrderedDict(baseline_sasa)
    HAS_sasa=OrderedDict(HAS_sasa)
    HAS_salt_bridge=OrderedDict(HAS_salt_bridge)
    ALL_salt_bridge=OrderedDict(ALL_salt_bridge)


    salt_bridge_aa=['E', 'D', 'K','H', 'R']
    n_pdbs_processed=0
    
    for ind,prot in tqdm(data.iterrows()):
        
        #!rm -r *.pdb
        #!rm -r *.gz
        
        # Build the command to remove all .pdb files
        import subprocess
        command = f'rm -r *.gz'
        subprocess.run(command, shell=True)
        command = f'rm -r *.pdb'
        subprocess.run(command, shell=True)

        id = prot.PDB_ID
        seq_mask =  (np.around(np.fromstring(prot.SEQ_MASK[1:-1], sep=' ')).astype(int)).tolist()
        seq_complete = prot.COMPLETE_SEQ
        seq_complete_aa=list(seq_complete) #reading aa per aa

        try:
            # 1. reading pdb
            pdb_id,chain=id.rsplit('_')
            pdb = prody.parsePDB(pdb_id,chain=chain)
            #prot = pdb.select('protein')
            pdb = pdb.select(f"protein and resid 1 to {pdb.select('pdbter').getResnums()[0]}")


            print('')
            print(f'Reading {pdb_id} with chain {chain}')
            print('')

            prody_seq=pdb.ca.getSequence()

            ####### SASA SCORE ##########

            try:
                mse = pdb.select('resname MSE')
                mse.setResnames(["MET"]*len(mse))
            except AttributeError:
                pass

            prody.writePDB(pdb_id+'_edit.pdb', pdb)
            parser = PDBParser()
            structure = parser.get_structure('pdb', pdb_id+'_edit.pdb')

            try:
                model = structure[0]
                dssp = DSSP(model, pdb_id+'_edit.pdb', dssp="dssp") #dssp='mkdssp'
                dssp_out = list(dssp)
                chain_dss= [lis for lis in dssp_out if lis]
                res_name = [lis[1] for lis in dssp_out]
                sasa_score = [lis[3] for lis in dssp_out]

                prody_seq=pdb.ca.getSequence()
                #print(prody_seq)

                #extract all the res id in the pdb
                res_id=pdb.ca.getResnums()

                #check
                if len(list(prody_seq)) != len(sasa_score):
                    print('--Error: length prody sequence and length sasa score are different')
                    continue
                if len(seq_complete_aa) != len(seq_mask):
                    print('--Error: length complete sequence and length seq masks are different')
                    break
                if len(list(prody_seq)) != len(list(seq_mask[i] for i in seq_mask if i == 1 )):
                    print('--Error: length prody sequence and length of mask == 1 are different')
                    break
                if len(list(res_id)) != len(list(seq_mask[i] for i in seq_mask if i == 1 )):
                    print('--Error: length res ids and length of mask == 1 are different')
                    break

                # SALT BRIDGE EXCRATION
                aa_id = pdb.select(acid_salt_bridges).getResnums()  #acid salt bridge aa id
                bb_id = pdb.select(basic_salt_bridges).getResnums() #basicsalt bridge aa id
                all_salt_bridge_ids = list(aa_id) + list(bb_id)
                #print(all_salt_bridge_ids)

                ### RUN THE MODEL ON SEQ_COMPLETE AND EXTRACT THE SASA SCORE ONLY FOR THE PARTIAL SEQUENCE FOR WHICH WE HAVE THE STRUCTURE (mask ==1)

                # computing attention respect to cls token
                df_all_vs_all,att_to_cls,df_att_to_cls_exp = attention_score_to_cls_token_and_to_all(seq_complete.replace('',' '),model_bert, device)
                att_to_cls = att_to_cls.drop(labels = ['[CLS]','[SEP]'])

                # creating all info df
                df = pd.DataFrame({'att_score': att_to_cls,'seq_mask':seq_mask})
                df = df.set_index([seq_complete_aa]) #res_name is for the prody seq
                # adding the sasa score only at aa with mask 1 (available in the pdb structure)
                df['sasa_score']=sasa_mask_selection(df['seq_mask'],sasa_score)
                #adding a res id column only at aa with mask 1 (available in the pdb structure)
                df['res_id']=sasa_mask_selection(df['seq_mask'],res_id)

                # creation of the BASELINE dict (baseline contains all the aminoacid so also the HAS, from the complete seq )
                for i in range(0,len(list(df.index.values))):
                    aa=df.index.values[i]
                    if float(df.sasa_score[i]) <= 0.25: # buried-not buried threshold after DSSP normalization
                        baseline_sasa[aa]['core'].append(1)
                        if (aa in salt_bridge_aa) == True:
                            if (df.res_id[i] in all_salt_bridge_ids) == True:  ## salt bridges checking for all E, D, K, H and R,
                                ALL_salt_bridge[aa]['core'].append(1)
                            else:
                                ALL_salt_bridge[aa]['core'].append(0)
                    else:
                        if float(df.sasa_score[i]) > 0.25:
                            baseline_sasa[aa]['surface'].append(1)
                            if (aa in salt_bridge_aa) == True:
                                if (df.res_id[i] in all_salt_bridge_ids) == True:  ## salt bridges checking for all E, D, K, H and R,
                                    ALL_salt_bridge[aa]['surface'].append(1)
                                else:
                                    ALL_salt_bridge[aa]['surface'].append(0)


                # extracting high attention score
                HAS_df=find_high_outliers_IQR(df)
                print('High attention df preview:')
                print(HAS_df)

                # creation of the HAS dict (from the complete seq)
                for i in range(0,len(list(HAS_df.index.values))):
                    aa=HAS_df.index.values[i]
                    #print(aa)
                    if float(HAS_df.sasa_score[i]) <= 0.25:
                        HAS_sasa[aa]['core'].append(1)
                        if (aa in salt_bridge_aa) == True:
                            if (HAS_df.res_id[i] in all_salt_bridge_ids) == True:  ## salt bridges checking for high  E, D, K, H and R,
                                HAS_salt_bridge[aa]['core'].append(1) # 1 if the HAS aa form a salt bridge, 0 if does not form salt bridge
                            else:
                                HAS_salt_bridge[aa]['core'].append(0)

                    else:
                        if float(HAS_df.sasa_score[i]) > 0.25:
                            HAS_sasa[aa]['surface'].append(1)
                            if (aa in salt_bridge_aa) == True:
                                if (HAS_df.res_id[i] in all_salt_bridge_ids) == True:  ## salt bridges checking for high  E, D, K, H and R,
                                    HAS_salt_bridge[aa]['surface'].append(1)
                                else:
                                    HAS_salt_bridge[aa]['surface'].append(0)


                # using the b-field as coloring method to check the threshold of sasa score (buried-not buried)
                if sasa_on_pdb==True:
                    import os
                    directory = './sasa_as_betas/'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    sasa_buried_not_buried=sasa_buried_not_buried_map(sasa_score)
                    if len(pdb.ca.getBetas()) != len(sasa_buried_not_buried):
                        print('--ERROR length betas is different from the length of sasa score')
                        break
                    pdb.setBetas(0)
                    pdb.ca.setBetas(sasa_buried_not_buried)
                    prody.writePDB((directory + pdb_id+'_'+chain+'_sasa_as_betas.pdb'),pdb.select('ca'))

                # using the b-field as coloring method to plot the attention score only in the aa present in the pdb while the score is obtained on the complete sequence
                if att_on_pdb==True:
                    import os
                    directory = './att_as_betas/'
                    if not os.path.exists(directory):
                        os.makedirs(directory) 
                    att_for_masked_aa = df[df['seq_mask'] == 1]['att_score'] * 100000
                    if len(pdb.ca.getBetas()) != len(att_for_masked_aa):
                        print('--ERROR length betas is different from the length of attention score')
                        break
                    pdb.setBetas(0)
                    pdb.ca.setBetas(att_for_masked_aa)
                    prody.writePDB((directory +pdb_id+'_'+chain+'_att_as_betas.pdb'),pdb.select('ca'))
                    if len(att_for_masked_aa) != len(prody_seq):
                        print('--ERROR inconsistency in the attention score length for masked aa and len of masked aa sequence (prody_seq')
                        break

                n_pdbs_processed+=1

            except (Bio.PDB.PDBExceptions.PDBException,Exception):
                pass
        except KeyError:
            pass
        except ValueError:
            pass

    return baseline_sasa,HAS_sasa, n_pdbs_processed , HAS_salt_bridge ,ALL_salt_bridge

#################################################################
################### Disulfide bonds selection ################### 
#################################################################

# Code source: Patrick Kunzmann
# License: BSD 3 clause

from tempfile import gettempdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb


def detect_disulfide_bonds(structure, distance=2.05, distance_tol=0.05,
                        dihedral=90, dihedral_tol=10):

    '''Explanation source: ChatGPT, code from https://www.biotite-python.org/examples/gallery/structure/disulfide_bonds.html
    This code is a Python function that detects disulfide bonds between cysteine residues in a protein structure. Disulfide bonds are covalent bonds between the sulfur atoms of two cysteine residues, which can help stabilize the protein structure.
    The function takes as input a structure object, which is assumed to be a BioPython Structure instance representing the protein structure, and several optional parameters:
        - distance: the expected distance between the sulfur atoms of the two cysteine residues involved in the bond (default is 2.05 Å).
        - distance_tol: a tolerance parameter for the distance (default is 0.05 Å).
        - dihedral: the expected dihedral angle between the two C-beta atoms of the cysteine residues (default is 90 degrees).
        - dihedral_tol: a tolerance parameter for the dihedral angle (default is 10 degrees).

    The function first identifies all the sulfur atoms in the protein structure that belong to cysteine residues.
    It then creates a cell list of all the sulfur atoms that are within a certain distance of each other, as specified by the distance and distance_tol parameters.
    This allows for efficient detection of potential bond partners. For each sulfur atom, the function iterates over all the potential bond partners within the cell list and checks whether the distance and dihedral angle
    criteria are met. If so, the indices of the two sulfur atoms involved in the bond are stored in a list and returned at the end of the function.'''

    # Array where detected disulfide bonds are stored
    disulfide_bonds = []
    # A mask that selects only S-gamma atoms of cysteins
    sulfide_mask = (structure.res_name == "CYS") & \
                (structure.atom_name == "SG")
    # sulfides in adjacency to other sulfides are detected in an
    # efficient manner via a cell list
    cell_list = struc.CellList(
        structure,
        cell_size=distance+distance_tol,
        selection=sulfide_mask
    )
    # Iterate over every index corresponding to an S-gamma atom
    for sulfide_i in np.where(sulfide_mask)[0]:
        # Find indices corresponding to other S-gamma atoms,
        # that are adjacent to the position of structure[sulfide_i]
        # We use the faster 'get_atoms_in_cells()' instead of
        # `get_atoms()`, as precise distance measurement is done
        # afterwards anyway
        potential_bond_partner_indices = cell_list.get_atoms_in_cells(
            coord=structure.coord[sulfide_i]
        )
        # Iterate over every index corresponding to an S-gamma atom
        # as bond partner
        for sulfide_j in potential_bond_partner_indices:
            if sulfide_i == sulfide_j:
                # A sulfide cannot create a bond with itself:
                continue
            # Create 'Atom' instances
            # of the potentially bonds S-gamma atoms
            sg1 = structure[sulfide_i]
            sg2 = structure[sulfide_j]
            # For dihedral angle measurement the corresponding
            # C-beta atoms are required, too
            cb1 = structure[
                (structure.chain_id == sg1.chain_id) &
                (structure.res_id == sg1.res_id) &
                (structure.atom_name == "CB")
            ]
            cb2 = structure[
                (structure.chain_id == sg2.chain_id) &
                (structure.res_id == sg2.res_id) &
                (structure.atom_name == "CB")
            ]
            # Measure distance and dihedral angle and check criteria
            bond_dist = struc.distance(sg1, sg2)
            bond_dihed = np.abs(np.rad2deg(struc.dihedral(cb1, sg1, sg2, cb2)))
            if bond_dist  > distance - distance_tol and \
                bond_dist  < distance + distance_tol and \
                bond_dihed > dihedral - dihedral_tol and \
                bond_dihed < dihedral + dihedral_tol:
                    # Atom meet criteria -> we found a disulfide bond
                    # -> the indices of the bond S-gamma atoms
                    # are put into a tuple with the lower index first
                    bond_tuple = sorted((sulfide_i, sulfide_j))
                    # Add bond to list of bonds, but each bond only once
                    if bond_tuple not in disulfide_bonds:
                        disulfide_bonds.append(bond_tuple)
    return np.array(disulfide_bonds, dtype=int)


def biotite_disulfide_bonds(data,model_bert,device):
    '''Arranged code by me to extract HAS and NON HAS aminoacids that are forming disulfide bonds, starting from pdb ids data.
    The code also perform the complete sasa analysis'''
    ## check code everything correct 26/01/2024
    #%shell
    #!gunzip *.gz

    from Bio.PDB import MMCIFParser,PDBParser
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.PDBList import PDBList
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.PDB.DSSP import DSSP
    import Bio.PDB.PDBExceptions
    import prody
    from collections import OrderedDict

    baseline_sasa={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}
    HAS_sasa={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}

    HAS_disulfide={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}
    ALL_disulfide={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}


    baseline_sasa=OrderedDict(baseline_sasa)
    HAS_sasa=OrderedDict(HAS_sasa)

    ALL_disulfide=OrderedDict(ALL_disulfide)
    HAS_disulfide=OrderedDict(HAS_disulfide)

    n_pdbs_processed=0
    for ind,prot in tqdm(data.iterrows()):
        
        # Build the command to remove all .pdb files
        import subprocess
        command = f'rm -r *.pdb.gz'
        subprocess.run(command, shell=True)
        command = f'rm -r *.pdb'
        subprocess.run(command, shell=True)

        id = prot.PDB_ID
        seq_mask =  (np.around(np.fromstring(prot.SEQ_MASK[1:-1], sep=' ')).astype(int)).tolist()
        seq_complete = prot.COMPLETE_SEQ
        seq_complete_aa=list(seq_complete) #reading aa per aa
        #print(seq_complete_aa)

        try:
            # 1. reading pdb
            pdb_id,chain=id.rsplit('_')
            pdb = prody.parsePDB(pdb_id,chain=chain)
            #prot = pdb.select('protein')
            pdb = pdb.select(f"protein and resid 1 to {pdb.select('pdbter').getResnums()[0]}")


            print('')
            print(f'Reading {pdb_id} with chain {chain}')
            print('')

            prody_seq=pdb.ca.getSequence()
            
            #print(prody_seq)

            ####### SASA SCORE ##########

            try:
                mse = pdb.select('resname MSE')
                mse.setResnames(["MET"]*len(mse))
            except AttributeError:
                pass

            prody.writePDB(pdb_id+'_edit.pdb', pdb)
            parser = PDBParser()
            structure = parser.get_structure('pdb', pdb_id+'_edit.pdb')
            
            try:
                model = structure[0]
                dssp = DSSP(model, pdb_id+'_edit.pdb', dssp="dssp") #dssp='mkdssp'
                dssp_out = list(dssp)
                chain_dss= [lis for lis in dssp_out if lis]
                res_name = [lis[1] for lis in dssp_out]
                sasa_score = [lis[3] for lis in dssp_out]

                prody_seq=pdb.ca.getSequence()
                #print('2',prody_seq)

                #extract all the res id in the pdb
                res_id=pdb.ca.getResnums()
                #print(res_id)

                #check
                if len(list(prody_seq)) != len(sasa_score):
                    print('--Error: length prody sequence and length sasa score are different')
                    continue
                if len(seq_complete_aa) != len(seq_mask):
                    print('--Error: length complete sequence and length seq masks are different')
                    break
                if len(list(prody_seq)) != len(list(seq_mask[i] for i in seq_mask if i == 1 )):
                    print('--Error: length prody sequence and length of mask == 1 are different')
                    break
                if len(list(res_id)) != len(list(seq_mask[i] for i in seq_mask if i == 1 )):
                    print('--Error: length res ids and length of mask == 1 are different')
                    break

                ### RUN THE MODEL ON SEQ_COMPLETE AND EXTRACT THE SASA SCORE ONLY FOR THE PARTIAL SEQUENCE FOR WHICH WE HAVE THE STRUCTURE (mask ==1)
                # computing attention respect to cls token
                df_all_vs_all,att_to_cls,df_att_to_cls_exp = attention_score_to_cls_token_and_to_all(seq_complete.replace('',' '),model_bert, device)
                att_to_cls = att_to_cls.drop(labels = ['[CLS]','[SEP]'])
                # creating all info df
                df = pd.DataFrame({'att_score': att_to_cls,'seq_mask':seq_mask})
                df = df.set_index([seq_complete_aa]) #res_name is for the prody seq
                #print(df)
                # adding the sasa score only at aa with mask 1 (available in the pdb structure)
                df['sasa_score']=sasa_mask_selection(df['seq_mask'],sasa_score)
                #adding a res id column only at aa with mask 1 (available in the pdb structure)
                df['res_id']=sasa_mask_selection(df['seq_mask'],res_id)

                # DISULFIDE BONDS EXTRACTION
                mmtf_file = mmtf.MMTFFile.read(rcsb.fetch(pdb_id, "mmtf", gettempdir())
                )
                knottin = mmtf.get_structure(mmtf_file, include_bonds=True, model=1)
                sulfide_indices = np.where((knottin.res_name == "CYS") & (knottin.atom_name == "SG"))[0]

                disulfide_bonds_ids=[] #disulfide bonds aa ids
                disulfide_bonds = detect_disulfide_bonds(knottin)

                for sg1_index, sg2_index in disulfide_bonds:
                    disulfide_bonds_ids.append(knottin[sg1_index].res_id)
                    disulfide_bonds_ids.append(knottin[sg2_index].res_id)
                print('DISULFIDE BONDS IDS',disulfide_bonds_ids)


                # creation of the BASELINE dict (baseline contains all the aminoacid so also the HAS, from the complete seq )
                for i in range(0,len(list(df.index.values))):
                    aa=df.index.values[i]
                    if float(df.sasa_score[i]) <= 0.25: # buried-not buried threshold after DSSP normalization
                        baseline_sasa[aa]['core'].append(1)
                        if (df.res_id[i] in disulfide_bonds_ids) == True:  ## disulfide bonds checking for all the C
                            ALL_disulfide[aa]['core'].append(1)
                        else:
                            ALL_disulfide[aa]['core'].append(0)
                    else:
                        if float(df.sasa_score[i]) > 0.25:
                            baseline_sasa[aa]['surface'].append(1)
                            if (df.res_id[i] in disulfide_bonds_ids) == True:  ## disulfide bonds checking for all the C
                                ALL_disulfide[aa]['surface'].append(1)
                            else:
                                ALL_disulfide[aa]['surface'].append(0)


                # extracting high attention score
                HAS_df=find_high_outliers_IQR(df)
                print('High attention df preview:')
                print(len(HAS_df))

                # creation of the HAS dict (from the complete seq)
                for i in range(0,len(list(HAS_df.index.values))):
                    aa=HAS_df.index.values[i]
                    #print(aa)
                    if float(HAS_df.sasa_score[i]) <= 0.25:
                        HAS_sasa[aa]['core'].append(1)

                        if (HAS_df.res_id[i] in disulfide_bonds_ids) == True:  ## disulfide bonds checking for HAS C
                            HAS_disulfide[aa]['core'].append(1)
                        else:
                            HAS_disulfide[aa]['core'].append(0)

                    else:
                        if float(HAS_df.sasa_score[i]) > 0.25:
                            HAS_sasa[aa]['surface'].append(1)

                        if (HAS_df.res_id[i] in disulfide_bonds_ids) == True:  ## disulfide bonds checking for HAS C
                            HAS_disulfide[aa]['surface'].append(1)
                        else:
                            HAS_disulfide[aa]['surface'].append(0)

                n_pdbs_processed+=1

            except (Bio.PDB.PDBExceptions.PDBException,Exception):
                print('bioerror')
                pass
        except KeyError:
            print('keyerror')
            pass
        except ValueError:
            print('valueerror')
            pass

    return baseline_sasa,HAS_sasa, n_pdbs_processed , HAS_disulfide, ALL_disulfide


def biopython_sasa_and_att_ca_coloring(data,sasa_on_pdb=False,att_on_pdb=False):
    #%shell
    #!gunzip *.gz
    
    ### CODICE PROBABILMENTE SBAGLIATO

    from Bio.PDB import MMCIFParser,PDBParser
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.PDBList import PDBList
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.PDB.DSSP import DSSP
    import Bio.PDB.PDBExceptions
    import prody
    from collections import OrderedDict

    baseline_sasa={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}
    HAS_sasa={'L':{'surface':[],'core':[]},'A':{'surface':[],'core':[]},'G':{'surface':[],'core':[]},'V':{'surface':[],'core':[]},'E':{'surface':[],'core':[]},'S':{'surface':[],'core':[]},'I':{'surface':[],'core':[]},'K':{'surface':[],'core':[]},'R':{'surface':[],'core':[]},'D':{'surface':[],'core':[]},'T':{'surface':[],'core':[]},'P':{'surface':[],'core':[]},'N':{'surface':[],'core':[]},'Q':{'surface':[],'core':[]},'F':{'surface':[],'core':[]},'Y':{'surface':[],'core':[]},'M':{'surface':[],'core':[]},'H':{'surface':[],'core':[]},'C':{'surface':[],'core':[]},'W':{'surface':[],'core':[]},}


    baseline_sasa=OrderedDict(baseline_sasa)
    HAS_sasa=OrderedDict(HAS_sasa)

    n_pdbs_processed=0

    for ind,prot in tqdm(data.iterrows()):

        #!rm -r *.pdb
        #!rm -r *.gz

        id = prot.PDB_ID
        seq_mask =  (np.around(np.fromstring(prot.SEQ_MASK[1:-1], sep=' ')).astype(int)).tolist()
        seq_complete = prot.COMPLETE_SEQ
        seq_complete_aa=list(seq_complete) #reading aa per aa

        try:
            # 1. reading pdb
            pdb_id,chain=id.rsplit('_')
            pdb = prody.parsePDB(pdb_id,chain=chain)
            #prot = pdb.select('protein')
            pdb = pdb.select(f"protein and resid 1 to {pdb.select('pdbter').getResnums()[0]}")

            print('')
            print(f'Reading {pdb_id} with chain {chain}')
            print('')

            prody_seq=pdb.ca.getSequence()

            ####### SASA SCORE ##########

            try:
                mse = pdb.select('resname MSE')
                mse.setResnames(["MET"]*len(mse))
            except AttributeError:
                pass

            prody.writePDB(pdb_id+'_edit.pdb', pdb)
            parser = PDBParser()
            structure = parser.get_structure('pdb', pdb_id+'_edit.pdb')

            try:
                model = structure[0]
                dssp = DSSP(model, pdb_id+'_edit.pdb', dssp="dssp") #dssp='mkdssp'
                dssp_out = list(dssp)
                chain_dss= [lis for lis in dssp_out if lis]
                res_name = [lis[1] for lis in dssp_out]
                sasa_score = [lis[3] for lis in dssp_out]

                prody_seq=pdb.ca.getSequence()
                #print(prody_seq)

                #extract all the res id in the pdb
                res_id=pdb.ca.getResnums()

                #check
                if len(list(prody_seq)) != len(sasa_score):
                    print('--Error: length prody sequence and length sasa score are different')
                    continue
                if len(seq_complete_aa) != len(seq_mask):
                    print('--Error: length complete sequence and length seq masks are different')
                    break
                if len(list(prody_seq)) != len(list(seq_mask[i] for i in seq_mask if i == 1 )):
                    print('--Error: length prody sequence and length of mask == 1 are different')
                    break
                if len(list(res_id)) != len(list(seq_mask[i] for i in seq_mask if i == 1 )):
                    print('--Error: length res ids and length of mask == 1 are different')
                    break

                ### RUN THE MODEL ON SEQ_COMPLETE AND EXTRACT THE SASA SCORE ONLY FOR THE PARTIAL SEQUENCE FOR WHICH WE HAVE THE STRUCTURE (mask ==1)

                # computing attention respect to cls token
                df_all_vs_all,att_to_cls,df_att_to_cls_exp = attention_score_to_cls_token_and_to_all(seq_complete.replace('',' '),model_bert,device)
                att_to_cls = att_to_cls.drop(labels = ['[CLS]','[SEP]'])

                # creating all info df
                df = pd.DataFrame({'att_score': att_to_cls,'seq_mask':seq_mask})
                df = df.set_index([seq_complete_aa]) #res_name is for the prody seq
                # adding the sasa score only at aa with mask 1 (available in the pdb structure)
                df['sasa_score']=sasa_mask_selection(df['seq_mask'],sasa_score)
                #adding a res id column only at aa with mask 1 (available in the pdb structure)
                df['res_id']=sasa_mask_selection(df['seq_mask'],res_id)

                # creation of the BASELINE dict (baseline contains all the aminoacid so also the HAS, from the complete seq )
                for i in range(0,len(list(df.index.values))):
                    aa=df.index.values[i]
                    if float(df.sasa_score[i]) <= 0.25: # buried-not buried threshold after DSSP normalization
                        baseline_sasa[aa]['core'].append(1)

                # extracting high attention score
                HAS_df=find_high_outliers_IQR(df)
                print('High attention df preview:')
                print(HAS_df)

                # creation of the HAS dict (from the complete seq)
                for i in range(0,len(list(HAS_df.index.values))):
                    aa=HAS_df.index.values[i]
                    #print(aa)
                    if float(HAS_df.sasa_score[i]) <= 0.25:
                        HAS_sasa[aa]['core'].append(1)

                # using the b-field as coloring method to check the threshold of sasa score (buried-not buried)
                if sasa_on_pdb==True:
                    sasa_buried_not_buried=sasa_buried_not_buried_map(sasa_score)
                    if len(pdb.ca.getBetas()) != len(sasa_buried_not_buried):
                        print('--ERROR length betas is different from the length of sasa score')
                        break
                    pdb.setBetas(0)
                    pdb.ca.setBetas(sasa_buried_not_buried)
                    prody.writePDB((pdb_id+'_'+chain+'_sasa_as_betas.pdb'),pdb.select('ca'))

                # using the b-field as coloring method to plot the attention score only in the aa present in the pdb while the score is obtained on the complete sequence
                if att_on_pdb==True:
                    att_for_masked_aa = df[df['seq_mask'] == 1]['att_score'] * 100000
                    if len(pdb.ca.getBetas()) != len(att_for_masked_aa):
                        print('--ERROR length betas is different from the length of attention score')
                        break
                    pdb.setBetas(0)
                    pdb.ca.setBetas(att_for_masked_aa)
                    prody.writePDB((pdb_id+'_'+chain+'_att_as_betas.pdb'),pdb.select('ca'))
                    if len(att_for_masked_aa) != len(prody_seq):
                        print('--ERROR inconsistency in the attention score length for masked aa and len of masked aa sequence (prody_seq')
                        break
                    print('PDB with attention score in beta field ready:',(pdb_id+'_'+chain+'_att_as_betas.pdb'))

                n_pdbs_processed+=1

            except (Bio.PDB.PDBExceptions.PDBException,Exception):
                pass
        except KeyError:
            pass
        except ValueError:
            pass

    return baseline_sasa,HAS_sasa, n_pdbs_processed
from analysis.attention_utils import *

red = (194/255, 22/255, 27/255, 1)
blue =( 24/255, 101/255, 172/255, 1) 

def fisher_exact_test_on_df(df,category):

    import numpy as np
    from scipy.stats import fisher_exact
    import pandas as pd
    # has
    conserved_high = len(df[(df['conserv_masks'] == True) & (df['att_outliers'] == True)])
    notconserved_high = len(df[(df['conserv_masks'] == False) & (df['att_outliers'] == True)])

    # non has
    conserved_low = len(df[(df['conserv_masks'] == True) & (df['att_outliers'] == False)])
    notconserved_low = len(df[(df['conserv_masks'] == False) & (df['att_outliers'] == False)])


    table=np.array([[conserved_high, conserved_low], [notconserved_high, notconserved_low]])
    df = pd.DataFrame(table, index=['conserved','not conserved'], columns=['high att','low att'])
    res = fisher_exact(table, alternative='two-sided')
    significance='SIGNIFICANT (p-value < 0.05)' if res.pvalue < 0.05 else 'non significant (p-value > 0.05)'
    with open(f'FISHER_exact.txt', 'a') as f:
        f.write(f'------------------------------------------------------ \n {category} - Fisher exact test \n------------------------------------------------------ \n')
        dfAsString = df.to_string(header=True, index=True)
        f.write(dfAsString+'\n'+'p-value: '+ str(res.pvalue)+'\n'+significance+'\n'+'\n')
        
    return True

def masking(conservation_masking,alignment_sequence):
    # Use list comprehension to filter both lists simultaneously
    masks, sequence = zip(*[(x, y) for x, y in zip(conservation_masking,alignment_sequence) if y != '-'])
    # Convert the filtered results back to lists
    masks = list(masks)
    sequence = list(sequence)
    return sequence, masks


def add_alignment_column(df, pattern):
    import pandas as pd
    aligned_rows = []  # List to hold the rows for the new DataFrame
    
    df_idx = 0  # Index to track position in the original DataFrame

    for char in pattern:
        if char == '-':
            # Add a dictionary representing a row with '-' or appropriate placeholders
            aligned_rows.append({'aa': '-', 'att_score': '-', 'conserv_masks': '-', 'att_outliers': '-', 'aln': '-'})
        else:
            # Find the matching row in the original DataFrame
            while df_idx < len(df) and df.iloc[df_idx]['aa'] != char:
                df_idx += 1
            
            if df_idx < len(df):
                # Copy the row and add it to the list, with the new column value set
                new_row = df.iloc[df_idx].to_dict()
                new_row['aln'] = char
                aligned_rows.append(new_row)
                df_idx += 1

    # Create a new DataFrame from the list of rows
    aligned_df = pd.DataFrame(aligned_rows, columns=df.columns.tolist() + ['aln'])
    return aligned_df


def alignments_masking_and_attention(thermo_seq,meso_seq,model,device):
    import numpy as np
    from Bio import pairwise2
    import pandas as pd
    from Bio.SubsMat import MatrixInfo as matlist
    matrix = matlist.blosum62
    # alignment between meso and thermo sequences
    #alignments = pairwise2.align.globalxx(thermo_seq,meso_seq,one_alignment_only=True) #len equal to each seq after masking function
    #aln = alignments[0] # with globalxx

    aln = pairwise2.align.globalds(thermo_seq,meso_seq, matrix, -0.5, -.1)[0]
    #aln = pairwise2.align.globalms(thermo_seq,meso_seq, 2, -1, -.5, -.1)[0] #len equal to each seq after masking function
    #aln = pairwise2.align.globaldx(meso_seq, thermo_seq, matrix)[0] #len longer then sequences after masking
    
    # when the aa is conserved between the 2 sequences 
    conservation_masking = np.array(list(aln.seqA))==np.array(list(aln.seqB))
    #alignment
    thermo_aln = list(aln.seqA)
    meso_aln = list(aln.seqB)
    # complete thermo df with a version also with alignment (also gaps)
    thermo_seq, thermo_masks = masking(conservation_masking,thermo_aln)
    thermo_att_score, thermo_att_outliers = attention_and_outliers(thermo_seq,model,device)
    thermo_df = pd.DataFrame({'aa':thermo_seq, 'att_score': thermo_att_score, 'conserv_masks':thermo_masks, 'att_outliers':thermo_att_outliers})
    thermo_df_aln = add_alignment_column(thermo_df,thermo_aln)
    # complete meso df  with a version also with alignment (also gaps)
    meso_seq, meso_masks = masking(conservation_masking,meso_aln)
    meso_att_score, meso_att_outliers = attention_and_outliers(meso_seq,model,device)
    
    meso_df = pd.DataFrame({'aa':meso_seq, 'att_score': meso_att_score, 'conserv_masks':meso_masks, 'att_outliers':meso_att_outliers})
    meso_df_aln = add_alignment_column(meso_df,meso_aln)


    return thermo_df,meso_df, thermo_att_score, meso_att_score, thermo_df_aln, meso_df_aln,conservation_masking


def plot_att_rostlab_vs_att_temberture(pdb_id,rostlab,temberture,df):
    import matplotlib.pyplot as plt
    # Creazione dello scatter plot
    fig = plt.figure(figsize=(4, 4))  # Imposta le dimensioni del grafico
    ax = fig.add_subplot(111, aspect='equal')
    
    # Define color and shape mappings
    color_map = {True: red, False: 'white'}
    shape_map = {True: '*', False: 'd'}  # Example shape mapping
    
    color = df['att_outliers'].map(color_map).tolist()
    shape = df['conserv_masks'].map(shape_map).tolist()

    # Aggiunta della linea diagonale
    limite = max(max(rostlab), max(temberture))


    for i in range(len(df)):
        plt.scatter(rostlab[i],temberture[i],  edgecolors='black', linewidth=0.5, color=color[i],marker=shape[i],s=60)
        
    
    plt.plot([0, limite], [0, limite],color='gray', linestyle='--',alpha=0.8,linewidth=3)  # Linea diagonale rossa tratteggiata
    ax.set_xlim(0,limite +0.001)
    plt.xlim(0,limite+0.001)
    
    # Imposta i ticks per l'asse x ogni 0.005
    plt.xticks(np.arange(0,limite +0.001, 0.005))

    # Imposta i ticks per l'asse y ogni 0.005, se necessario
    plt.yticks(np.arange(0,limite +0.001, 0.005))
    # Aggiunta di titolo e legenda
    plt.title(f'{pdb_id}')
    plt.xlabel('Attention score from prot_bert_bfd')
    plt.ylabel('Attention score from TemBERTure$_{CLS}$')
    
    #ticks =  ax.get_yticks()
    #ax.set_yticklabels([("{0:.3f}".format((tick))) for tick in ticks],fontsize=12)
    #ticks =  ax.get_xticks()
    #ax.set_xticklabels([("{0:.3f}".format((tick))) for tick in ticks],fontsize=12)
    
    ax.spines['bottom'].set_linewidth(2)  # Spessore dell'asse x
    ax.spines['left'].set_linewidth(2)    # Spessore dell'asse y
    ax.spines['top'].set_linewidth(0)  # Spessore dell'asse x
    ax.spines['right'].set_linewidth(0)    # Spessore dell'asse y

    ax.tick_params(axis='x', labelsize=12, length=10, width=2) 
    ax.tick_params(axis='y', labelsize=12, length=10, width=2)
    
    plt.tight_layout()
    plt.savefig(f'rostlab_vs_temberture/single/{pdb_id}.png',format='png', transparent=True)
    plt.savefig(f'rostlab_vs_temberture/single/{pdb_id}.svg',format='svg', transparent=True)


def plot_thermo_vs_meso_has_alignment(pdb_id_thermo, pdb_id_meso, aln_df):
    import matplotlib.pyplot as plt
    import numpy as np

    # keeping only the conserved aa 
    aln_df['att_outliers'] = aln_df.apply(
        lambda row: 'True_both' if row['thermo_att_outliers'] is True and row['meso_att_outliers'] is True 
        else ('True_thermo' if row['thermo_att_outliers'] is True 
        else ('True_meso' if row['meso_att_outliers'] is True 
        else 'False')), axis=1)
    #aln_df['conserved_aa'] = aln_df['thermo_conserv_masks'].apply(lambda x: False if x in [False, '-'] else True)
    #aln_df['conserved_aa'] = aln_df['thermo_conserv_masks'].apply(lambda x: 'gap' if x == '-' else (False if x is False else True))
    aln_df['conserved_aa'] = aln_df.apply(lambda row: 'gap' if row['thermo_conserv_masks'] == '-' or row['meso_conserv_masks'] == '-' else (False if row['thermo_conserv_masks'] is False else True), axis=1)
    aln_df['meso_att_score'] = aln_df['meso_att_score'].apply(lambda x: 0 if x in [False, '-'] else x)
    aln_df['thermo_att_score'] = aln_df['thermo_att_score'].apply(lambda x: 0 if x in [False, '-'] else x)
    #keeping all the aa except for the gaps only the one that are has
    aln_df = aln_df[(aln_df['conserved_aa'] != 'gap') | ((aln_df['conserved_aa'] == 'gap') & aln_df['att_outliers'].isin(['True_meso', 'True_thermo','True_both']))]
    
    # Inizializzazione del grafico scatter
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    
    # Aggiunta della linea diagonale
    limite = max(pd.to_numeric(aln_df['meso_att_score'], errors='coerce').max(),pd.to_numeric(aln_df['thermo_att_score'], errors='coerce').max())
    ax.plot([0, limite], [0, limite], color='gray', linestyle='--', alpha=0.8, linewidth=3)
    ax.set_xlim(0, limite + 0.001)
    ax.set_ylim(0, limite + 0.001)
    

    # Definizione dei colori e delle forme
    color_map = {'True_thermo': red, 'True_meso': blue, 'False': 'white'}
    shape_map = {True: 'o', False: 'd', 'gap': '^'}
    
    marker_style = dict(c=blue, linestyle=':',fillstyle='left',markerfacecoloralt=red,markeredgecolor='black')
    
    # Plot dei punti
    for i, row in aln_df.iterrows():
        if row['att_outliers'] == 'True_both':
            marker = shape_map[row['conserved_aa']]
            # Disegna due semi-cerchi per i punti 'True_both'
            ax.plot(row['meso_att_score'], row['thermo_att_score'],  linewidth=0.01, markersize=8, marker=marker, **marker_style, clip_on=False, alpha=0.7)
        else:
            # Plot normale per gli altri punti
            color = color_map[row['att_outliers']]
            marker = shape_map[row['conserved_aa']]
            ax.plot(row['meso_att_score'], row['thermo_att_score'],  linewidth=0.01, color=color, markersize=8, marker=marker,markeredgecolor='black', clip_on=False, alpha=0.7)


    
    from matplotlib.ticker import MaxNLocator
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.gca().xaxis.get_majorticklabels()[0].set_x(-10)
    
    def format_x(tick_val, tick_pos):
        if tick_val == 0:
            return f"{int(tick_val)}"
        else:
            return f"{tick_val}"

    import matplotlib.ticker as ticker
    # Applica la formattazione personalizzata all'asse x
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_x))


    # Impostazione dei tick
    ax.set_xticks(np.arange(0, limite + 0.001, 0.005))
    ax.set_yticks(np.arange(0, limite + 0.001, 0.005))
    
    meso = pdb_id_meso.split('_')[0]
    thermo = pdb_id_thermo.split('_')[0]

    # Aggiunta di titolo e legenda
    plt.title(f'{meso} - {thermo} ')
    plt.xlabel('Non-thermophilic att.score',fontsize=15)
    plt.ylabel('Thermophilic att.score',fontsize=15)

    # Impostazioni delle linee del grafico
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.tick_params(axis='x', labelsize=14, length=10, width=2)
    ax.tick_params(axis='y', labelsize=14, length=10, width=2)
    
    
    # Salvataggio del grafico
    plt.tight_layout()
    plt.savefig(f'temberture/thermo_vs_meso_aa_conserved/{pdb_id_thermo}_{pdb_id_meso}.png', format='png', )
    plt.savefig(f'temberture/thermo_vs_meso_aa_conserved/{pdb_id_thermo}_{pdb_id_meso}.svg', format='svg', )


    return aln_df


def plot_thermo_vs_meso_has_alignment_main_figure(pdb_id_thermo, pdb_id_meso, aln_df):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Inizializzazione del grafico scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    # Aggiunta della linea diagonale
    limite = max(pd.to_numeric(aln_df['meso_att_score'], errors='coerce').max(),pd.to_numeric(aln_df['thermo_att_score'], errors='coerce').max())
    ax.plot([0, limite], [0, limite], color='gray', linestyle='--', alpha=0.8, linewidth=3)
    ax.set_xlim(0, limite + 0.001)
    ax.set_ylim(0, limite + 0.001)
    

    # keeping only the conserved aa 
    aln_df['att_outliers'] = aln_df.apply(
        lambda row: 'True_both' if row['thermo_att_outliers'] is True and row['meso_att_outliers'] is True 
        else ('True_thermo' if row['thermo_att_outliers'] is True 
        else ('True_meso' if row['meso_att_outliers'] is True 
        else 'False')), axis=1)
    #aln_df['conserved_aa'] = aln_df['thermo_conserv_masks'].apply(lambda x: False if x in [False, '-'] else True)
    #aln_df['conserved_aa'] = aln_df['thermo_conserv_masks'].apply(lambda x: 'gap' if x == '-' else (False if x is False else True))
    aln_df['conserved_aa'] = aln_df.apply(lambda row: 'gap' if row['thermo_conserv_masks'] == '-' or row['meso_conserv_masks'] == '-' else (False if row['thermo_conserv_masks'] is False else True), axis=1)
    aln_df['meso_att_score'] = aln_df['meso_att_score'].apply(lambda x: 0 if x in [False, '-'] else x)
    aln_df['thermo_att_score'] = aln_df['thermo_att_score'].apply(lambda x: 0 if x in [False, '-'] else x)
    #keeping all the aa except for the gaps only the one that are has
    aln_df = aln_df[(aln_df['conserved_aa'] != 'gap') | ((aln_df['conserved_aa'] == 'gap') & aln_df['att_outliers'].isin(['True_meso', 'True_thermo','True_both']))]
    

    # Definizione dei colori e delle forme
    color_map = {'True_thermo': red, 'True_meso': blue, 'False': 'white'}
    shape_map = {True: 'o', False: 'd', 'gap': '^'}
    
    marker_style = dict(c=blue, linestyle=':',fillstyle='left',markerfacecoloralt=red,markeredgecolor='black')
    
    # Plot dei punti
    for i, row in aln_df.iterrows():
        if row['att_outliers'] == 'True_both':
            marker = shape_map[row['conserved_aa']]
            # Disegna due semi-cerchi per i punti 'True_both'
            ax.plot(row['meso_att_score'], row['thermo_att_score'],  linewidth=0.01, markersize=13, marker=marker, **marker_style, clip_on=False, alpha=0.7)
        else:
            # Plot normale per gli altri punti
            color = color_map[row['att_outliers']]
            marker = shape_map[row['conserved_aa']]
            ax.plot(row['meso_att_score'], row['thermo_att_score'],  linewidth=0.01, color=color, markersize=13, marker=marker,markeredgecolor='black', clip_on=False, alpha=0.7)



    from matplotlib.ticker import MaxNLocator
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    
    def format_x(tick_val, tick_pos):
        if tick_val == 0:
            return f"{int(tick_val)}"
        else:
            return f"{tick_val}"

    import matplotlib.ticker as ticker
    # Applica la formattazione personalizzata all'asse x
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_x))

    # Impostazione dei tick
    ax.set_xticks(np.arange(0, limite + 0.001, 0.002))
    ax.set_yticks(np.arange(0, limite + 0.001, 0.002))
    
    meso = pdb_id_meso.split('_')[0]
    thermo = pdb_id_thermo.split('_')[0]

    # Aggiunta di titolo e legenda
    plt.xlabel('1LDG attention score',fontsize=18) #non thermophilic
    plt.ylabel('1LDN attention score',fontsize=18) # thermophilic

    # Impostazioni delle linee del grafico
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.tick_params(axis='x', labelsize=14, length=10, width=2)
    ax.tick_params(axis='y', labelsize=14, length=10, width=2)
    
    
    # Salvataggio del grafico
    plt.tight_layout()
    plt.savefig(f'./entropy/{pdb_id_thermo}_{pdb_id_meso}.png', format='png', transparent=True)
    plt.savefig(f'./entropy/{pdb_id_thermo}_{pdb_id_meso}.svg', format='svg', transparent=True)


    return aln_df
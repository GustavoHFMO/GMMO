'''
Created on 17 de set de 2018
@author: gusta
'''

from sklearn.cross_validation import StratifiedKFold
from competitive_algorithms.IGMM_CD import IGMMCD
from table_generator.excel_table import Tabela_excel
from streams.readers.arff_reader import ARFFReader
from data_streams.static_datasets import Datasets
from sklearn.metrics import accuracy_score
import matplotlib.patches as patches
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import array
import seaborn as sns
import pandas as pd
import numpy as np
plt.style.use('seaborn-whitegrid')

       
def main():
    
    # experimental setup
    datasets = ['elecNormNew', 
                'FG_2C_2D',
                'GEARS_2C_2D',
                'keystroke',
                'MG_2C_2D',
                'UG_2C_2D',
                'UG_2C_3D',
                'UG_2C_5D']
    T = [1, 13, 13, 7, 13, 13, 13, 13]

    # store the final accuracy
    tb_accuracy = Tabela_excel()
    tb_accuracy.Criar_tabela(nome_tabela='IGMM-CD-accuracy', 
                                 folhas=['IGMM-CD'], 
                                 cabecalho=datasets, 
                                 largura_col=5000)
    
    # running the code
    for i in range(3, len(datasets)):    
        labels, attributes, stream_records = ARFFReader.read("../data_streams/bracis2015/"+datasets[i]+".arff")
        igmmcd = IGMMCD(2, 0.01, T[i])
        igmmcd.NAME = datasets[i]
        igmmcd.prequential(labels, stream_records)
        
        # storing only the predictions
        df = pd.DataFrame(data={'predictions': igmmcd.PREDICTIONS})
        df.to_csv("IGMM-CD-"+datasets[i]+".csv")
            
        # storing the accuracy of system 
        tb_accuracy.Adicionar_dado(0, 1, i, igmmcd.accuracyGeneral())
        
if __name__ == "__main__":
    main()        

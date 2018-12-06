'''
Created on 5 de out de 2018
@author: gusta
'''

from streams.readers.arff_reader import ARFFReader
from tasks.prequential import Prequential
from detectors.ewma import EWMA
from gaussian_models.kdnagmm import KDNAGMM
from competitive_algorithms.IGMM_CD import IGMMCD
from competitive_algorithms.Dynse import PrunningEngine
from competitive_algorithms.Dynse import ClassificationEngine
from competitive_algorithms.Dynse import Dynse
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from table_generator.excel_table import Tabela_excel

def chooseDataset(number, variation):
    if(number==0):
        labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/circles/circles_"+str(variation)+".arff")
        name = 'circles'
    
    elif(number==1):
        labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine1/sine1_"+str(variation)+".arff")
        name = 'sine1'
    
    elif(number==2):
        labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine2/sine2_"+str(variation)+".arff")
        name = 'sine2'
    
    elif(number==3):
        labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/virtual_5changes/virtual_5changes_"+str(variation)+".arff")
        name = 'virtual_5changes'
    
    elif(number==4):
        labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/virtual_9changes/virtual_9changes_"+str(variation)+".arff")
        name = 'virtual_9changes'
    
    elif(number==5):
        labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/SEA/SEA_"+str(variation)+".arff")
        name = 'SEA'
    
    elif(number==6):
        labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/SEARec/SEARec_"+str(variation)+".arff")
        name = 'SEARec'
    
    elif(number==7):
        labels, _, stream_records = ARFFReader.read("data_streams/real/noaa.arff")
        name = 'noaa'
            
    elif(number==8):
        labels, _, stream_records = ARFFReader.read("data_streams/real/elec.arff")
        name = 'elec'
        
    elif(number==9):
        labels, _, stream_records = ARFFReader.read("data_streams/real/PAKDD.arff")
        name = 'PAKDD'
    
    print(name)
    return name, labels, stream_records
    
def saveInformation(i, xxx, models, name, tb_accuracy, predictions, target, accuracy):
    
    # storing the prediction
    #tb_predictions.Adicionar_coluna(xxx, i, predictions)
    
    df = pd.DataFrame(data={'predictions': predictions, 'target': target})
    df.to_csv(patch+models[xxx]+"-"+name+"_"+str(i)+".csv")
        
    # storing the accuracy of system 
    tb_accuracy.Adicionar_dado(0, i+1, xxx, accuracy)
    print(models[xxx], ': ', accuracy)
    

# models name
models = ['Proposed Method', 'IGMM-CD', 'Dynse-priori']

window_size = 100
executions = 30

# parameters
patch = "projects/competitive_real/"

for i in range(9, 10):
    
    for j in range(executions):
        
        # choosing the dataset
        name, labels, stream_records = chooseDataset(i, j)

        # table to store only the accuracy of models        
        if(j == 0):
            tb_accuracy = Tabela_excel()
            tb_accuracy.Criar_tabela(nome_tabela=patch+name+'-accuracy', 
                                         folhas=['modelos'], 
                                         cabecalho=models, 
                                         largura_col=5000)

        ################################################################################## 0 #######################################################
        # Proposed 
        xxx = 0
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=True, kmax=2)
        
        #2. instantiate the detector
        detector = EWMA(min_instance=window_size, c=1, w=0.5)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=window_size)
        
        #4. execute the prequential
        g.run_cross_validation(j, executions)
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################

        ################################################################################## 1 #######################################################
        # IGMM-CD
        xxx = 1
            
        #1. import the classifier
        igmmcd = IGMMCD(10, 0.01, 9)
        
        #2. execute the prequential
        igmmcd.prequential_cross_validation(labels, stream_records, window_size, j, executions)
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, igmmcd.returnPredictions(), igmmcd.returnTarget(), igmmcd.accuracyGeneral())
        ############################################################################################################################################

        ################################################################################## 2 #######################################################
        # Dynse
        xxx = 2
            
        #1. instanciando o mecanismo de classificacao
        ce = ClassificationEngine('priori')
     
        #2. definindo o criterio de poda
        pe = PrunningEngine('age')
           
        #3. instanciando o classificador base
        bc = GaussianNB()
        
        #4. instanciando o framework
        dynse = Dynse(D=25,
                      M=4, 
                      K=5, 
                      CE=ce, 
                      PE=pe, 
                      BC=bc)
         
        #5. executando o framework
        dynse.prequential_cross_validation(labels=labels, 
                          stream=stream_records,
                          train_size=50,
                          window_size=window_size,
                          fold=j,
                          qtd_folds=executions)
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, dynse.returnPredictions(), dynse.returnTarget(), dynse.accuracyGeneral())
        ############################################################################################################################################

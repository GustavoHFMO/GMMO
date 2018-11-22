'''
Created on 5 de out de 2018
@author: gusta
'''

from table_generator.excel_table import Tabela_excel
from streams.readers.arff_reader import ARFFReader
from detectors.no_detector import noDetector
from gaussian_models.kdnagmm import KDNAGMM
from tasks.prequential import Prequential
from detectors.ewma import EWMA
import pandas as pd
import numpy as np

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
    

# parametros estaticos
train_size_estatico = 50
Kmax_estatico = 5
lt = 1
patch = "projects/analiseParametros/"

# parametros a serem analisados
train_size = [100, 200, 300, 400]
Kmax = [1, 3, 7, 9]

# nomes dos modelos gerados
models = []
for k in range(len(train_size)):
    nome = "train_size:"+str(train_size[k])+"-kmax:"+str(Kmax_estatico)
    models.append(nome)
for k in range(len(Kmax)):
    nome = "train_size:"+str(train_size_estatico)+"-kmax:"+str(Kmax[k])
    models.append(nome)


for i in range(0, 7):
    
    for j in range(0, 10):
        
        # choosing the dataset
        name, labels, stream_records = chooseDataset(i, j)

        # table to store only the accuracy of models        
        if(j == 0):
            tb_accuracy = Tabela_excel()
            tb_accuracy.Criar_tabela(nome_tabela=patch+name+'-accuracy', 
                                         folhas=['modelos'], 
                                         cabecalho=models, 
                                         largura_col=5000)
            
        
        # to initiate the models
        xxx = 0
        
        # analise de sensitividade sobre o tamanho do treinamento
        for k in range(0, len(train_size)):
        
            ################################################################################## 0 #######################################################
            xxx += k
                
            #1. import the classifier
            classifier = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=False, kmax=Kmax_estatico)
            classifier.NAME = "train_size:"+str(train_size[k])+"-kmax:"+str(Kmax_estatico)
            
            #2. instantiate the detector
            detector = EWMA(min_instance=train_size[k], lt=lt)
            
            #3. instantiate the prequetial
            g = Prequential(name=classifier.NAME,
                            labels=labels,
                            stream=stream_records,
                            classifier=classifier,
                            detector=detector,
                            strategy=True,
                            window_size=train_size[k])
            
            #4. execute the prequential
            g.run()
            
            #5. storing the information
            saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
            ############################################################################################################################################
            
        
        # analise de sensitividade sobre o kmax
        for k in range(0, len(Kmax)):
        
            ################################################################################## 0 #######################################################
            xxx += k
                
            #1. import the classifier
            classifier = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=False, kmax=Kmax[k])
            classifier.NAME = "train_size:"+str(train_size_estatico)+"-kmax:"+str(Kmax[k])
            
            #2. instantiate the detector
            detector = EWMA(min_instance=train_size_estatico, lt=lt)
            
            #3. instantiate the prequetial
            g = Prequential(name=classifier.NAME,
                            labels=labels,
                            stream=stream_records,
                            classifier=classifier,
                            detector=detector,
                            strategy=True,
                            window_size=train_size_estatico)
            
            #4. execute the prequential
            g.run()
            
            #5. storing the information
            saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
            ############################################################################################################################################

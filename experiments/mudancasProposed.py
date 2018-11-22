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
    

# models name
models = ['com ruido', 
          'sem remocao', 
          'sem adicao', 
          'tudo', 
          'nada', 
          'atualizacao constante', 
          'atualizacao constante sem remocao', 
          'sem deteccao'
          'atualizacao n steps',
          'virtual-ruidos']

# parameters
train_size = 50
lt = 1
patch = "projects/modificacoes2/"

for i in range(0, 4):
    
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
            
        
        '''
        ################################################################################## 0 #######################################################
        # ruido
        xxx = 0
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=False, remocao=True, adicao=True, erro=False)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
    
        ################################################################################## 1 #######################################################
        # remocao
        xxx = 1
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=False, adicao=True, erro=False)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
        
        ################################################################################## 2 #######################################################
        # adicao
        xxx = 2
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=True, adicao=False, erro=False)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
        
        ################################################################################## 3 #######################################################
        # tudo
        xxx = 3
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=False)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
        
        ################################################################################## 4 #######################################################
        # nada
        xxx = 4
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=False, remocao=False, adicao=False, erro=False)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=False,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
    
        ################################################################################## 5 #######################################################
        # atualizacao sem erro
        xxx = 5
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=True)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
        
        ################################################################################## 6 #######################################################
        # atualizacao sem erro e remocao
        xxx = 6
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=False, adicao=True, erro=True)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
        '''
            
        ################################################################################## 7 #######################################################
        #atualizacao n steps
        xxx = 7
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=True)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        #detector = EWMA(min_instance=train_size, lt=lt)
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=True,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
        
        ################################################################################## 8 #######################################################
        #virtual-ruidos
        xxx = 8
            
        #1. import the classifier
        classifier = KDNAGMM(ruido=True, remocao=False, adicao=False, erro=False)
        classifier.NAME = models[xxx]
        
        #2. instantiate the detector
        detector = EWMA(min_instance=train_size, lt=lt)
        
        #3. instantiate the prequetial
        g = Prequential(name=models[xxx],
                        labels=labels,
                        stream=stream_records,
                        classifier=classifier,
                        detector=detector,
                        strategy=False,
                        window_size=train_size)
        
        #4. execute the prequential
        g.run()
        
        #5. storing the information
        saveInformation(j, xxx, models, name, tb_accuracy, g.returnPredictions(), g.returnTarget(), g.accuracyGeneral())
        ############################################################################################################################################
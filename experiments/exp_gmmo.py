'''
Created on 6 de set de 2018
@author: gusta
'''

from streams.readers.arff_reader import ARFFReader
from tasks.prequential import Prequential
from detectors.ewma import EWMA
from detectors.no_detector import noDetector
from classifiers.gmm import GaussianMixture
from classifiers.agmm import AGMM
import pandas as pd
from table_generator.excel_table import Tabela_excel

def chooseDataset(number):
    if(number==0):
        _, _, stream_records = ARFFReader.read("data_streams/_synthetic/circles.arff")
        name = 'circles'
        print(name)
        return name, stream_records
    
    elif(number==1):
        _, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine1.arff")
        name = 'sine1'
        print(name)
        return name, stream_records
    
    elif(number==2):
        _, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine2.arff")
        name = 'sine2'
        print(name)
        return name, stream_records
    
    elif(number==3):
        _, _, stream_records = ARFFReader.read("data_streams/_synthetic/virtual.arff")
        name = 'virtual'
        print(name)
        return name, stream_records

def saveInformation(i, xxx, models, name, tb_accuracy, predictions, accuracy):
    
    # storing the prediction
    #tb_predictions.Adicionar_coluna(xxx, i, predictions)
    
    df = pd.DataFrame(data={'predictions': predictions})
    df.to_csv(models[xxx]+"-"+name+".csv")
        
    # storing the accuracy of system 
    tb_accuracy.Adicionar_dado(0, i+1, xxx, accuracy)
    print(models[xxx], ': ', accuracy)
    

# models name
models = ['GMM', 'GMMO', 'GMM+ECDD', 'GMMO+ECDD',
           'AGMMO', 'AGMM+ECDD', 'AGMMO+ECDD']

# parameters
m = 300
batch = 250
lt = 1

for i in range(3, 4):
    
    # choosing the dataset
    name, stream_records = chooseDataset(i)
    
    '''
    # table to store the predictions of models
    tb_predictions = Tabela_excel()
    tb_predictions.Criar_tabela(nome_tabela=name+'-predictions', 
                    folhas=models, 
                    cabecalho=None, 
                    largura_col=5000)
    '''
     
    # table to store only the accuracy of models 
    tb_accuracy = Tabela_excel()
    tb_accuracy.Criar_tabela(nome_tabela=name+'-accuracy', 
                                 folhas=['modelos'], 
                                 cabecalho=models, 
                                 largura_col=5000)
    
    
    ################################################################################## 1 #######################################################
    # GMM
    xxx = 0
        
    #2. import the classifier
    gmm = GaussianMixture()
    
    #3. instantiate the detector
    nodetect = noDetector()
    
    #4. instantiate the prequetial
    a = Prequential(name=models[xxx],
                    stream=stream_records,
                              classifier=gmm,
                              detector=nodetect,
                              strategy=False,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    a.run()
    
    #6. storing the information
    saveInformation(0, xxx, models, name, tb_accuracy, a.returnPredictions(), a.accuracyGeneral())
    ############################################################################################################################################    
    
    ################################################################################## 2 #######################################################
    # GMMO
    xxx = 1
        
    #2. import the classifier
    gmm = GaussianMixture()
    
    #3. instantiate the detector
    nodetect = noDetector()
    
    #4. instantiate the prequetial
    b = Prequential(name=models[xxx],
                    stream=stream_records,
                              classifier=gmm,
                              detector=nodetect,
                              strategy=True,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    b.run()
    
    #6. storing the information
    saveInformation(0, xxx, models, name, tb_accuracy, b.returnPredictions(), b.accuracyGeneral())
    ############################################################################################################################################
    
    ################################################################################## 3 #######################################################
    # GMM+ECDD
    xxx = 2
        
    #2. import the classifier
    gmm = GaussianMixture()
    
    #3. instantiate the detector
    ewma = EWMA(min_instance=m, lt=lt)
    
    #4. instantiate the prequetial
    c = Prequential(name=models[xxx],
                    stream=stream_records,
                              classifier=gmm,
                              detector=ewma,
                              strategy=False,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    c.run()
    
    #6. storing the information
    saveInformation(0, xxx, models, name, tb_accuracy, c.returnPredictions(), c.accuracyGeneral())
    ############################################################################################################################################
    
    
    ################################################################################## 3 #######################################################
    # GMM0+ECDD
    xxx = 3
        
    #2. import the classifier
    gmm = GaussianMixture()
    
    #3. instantiate the detector
    ewma = EWMA(min_instance=m, lt=lt)
    
    #4. instantiate the prequetial
    d = Prequential(name=models[xxx],
                    stream=stream_records,
                              classifier=gmm,
                              detector=ewma,
                              strategy=True,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    d.run()
    
    #6. storing the information
    saveInformation(0, xxx, models, name, tb_accuracy, d.returnPredictions(), d.accuracyGeneral())
    ############################################################################################################################################

    ################################################################################## 4 #######################################################
    # AGMMO
    xxx = 4
        
    #2. import the classifier
    agmm = AGMM()
    
    #3. instantiate the detector
    nodetect = noDetector()
    
    #4. instantiate the prequetial
    e = Prequential(name=models[xxx],
                    stream=stream_records,
                              classifier=agmm,
                              detector=nodetect,
                              strategy=True,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    e.run()
    
    #6. storing the information
    saveInformation(0, xxx, models, name, tb_accuracy, e.returnPredictions(), e.accuracyGeneral())
    ############################################################################################################################################
    
    ################################################################################## 5 #######################################################
    # AGMM+ECDD
    xxx = 5
        
    #2. import the classifier
    agmm = AGMM()
    
    #3. instantiate the detector
    ewma = EWMA(min_instance=m, lt=lt)
    
    #4. instantiate the prequetial
    f = Prequential(name=models[xxx],
                    stream=stream_records,
                              classifier=agmm,
                              detector=ewma,
                              strategy=False,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    f.run()
    
    #6. storing the information
    saveInformation(0, xxx, models, name, tb_accuracy, f.returnPredictions(), f.accuracyGeneral())
    ############################################################################################################################################
    
    
    ################################################################################## 6 #######################################################
    # AGMM0+ECDD
    xxx = 6
        
    #2. import the classifier
    agmm = AGMM()
    
    #3. instantiate the detector
    ewma = EWMA(min_instance=m, lt=lt)
    
    #4. instantiate the prequetial
    g = Prequential(name=models[xxx],
                    stream=stream_records,
                              classifier=agmm,
                              detector=ewma,
                              strategy=True,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    g.run()
    
    #6. storing the information
    saveInformation(0, xxx, models, name, tb_accuracy, g.returnPredictions(), g.accuracyGeneral())
    ############################################################################################################################################
    
    
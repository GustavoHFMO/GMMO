'''
Created on 5 de out de 2018
@author: gusta
'''

from table_generator.excel_table import Tabela_excel
from streams.readers.arff_reader import ARFFReader
from competitive_algorithms.IGMM_CD import IGMMCD
from competitive_algorithms.Dynse import PrunningEngine
from competitive_algorithms.Dynse import ClassificationEngine
from competitive_algorithms.Dynse import Dynse
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import warnings
warnings.simplefilter("ignore", UserWarning)

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
patch = "projects/analiseParametrosCompetitive/"

# parametros a serem analisados
train_size = [100, 200, 300, 400]
Kmax = [1, 3, 7, 9]

# nomes dos modelos gerados
models = []
for k in range(len(train_size)):
    nome = "Dynse-train_size"+str(train_size[k])
    models.append(nome)
for k in range(len(Kmax)):
    nome = "IGMM-CD-kmax"+str(Kmax[k])
    models.append(nome)


for i in range(4, 7):
    
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
        for k in range(len(train_size)):
        
            ################################################################################## 0 #######################################################
            #1. import the classifier
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
            dynse.prequential(labels=labels, 
                              stream=stream_records, 
                              window_size=100,
                              train_size=train_size[k])

            
            #5. storing the information
            saveInformation(j, xxx, models, name, tb_accuracy, dynse.returnPredictions(), dynse.returnTarget(), dynse.accuracyGeneral())
            
            xxx += 1
            ############################################################################################################################################
        
        # analise de sensitividade sobre o kmax
        for k in range(len(Kmax)):
        
            ################################################################################## 0 #######################################################
            #1. import the classifier
            igmmcd = IGMMCD(0.5, 0.01, Kmax[k])
            
            #2. execute the prequential
            igmmcd.prequential(labels, stream_records, train_size_estatico)
            
            #5. storing the information
            saveInformation(j, xxx, models, name, tb_accuracy, igmmcd.returnPredictions(), igmmcd.returnTarget(), igmmcd.accuracyGeneral())
            
            xxx += 1
            ############################################################################################################################################

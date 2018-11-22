'''
Created on 6 de set de 2018
@author: gusta
'''
from data_streams.adjust_labels import Adjust_labels
al = Adjust_labels()
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#print(plt.style.available)

def chooseDataset(number, variation):
    if(number==0):
        name = 'circles'
        name_variation = name+'_'+str(variation)
        drifts = [2000, 4000, 6000]
    
    elif(number==1):
        name = 'sine1'
        name_variation = name+'_'+str(variation)
        drifts = [2000, 4000, 6000, 8000]
    
    elif(number==2):
        name = 'sine2'
        name_variation = name+'_'+str(variation)
        drifts = [2000, 4000, 6000, 8000]
    
    elif(number==3):
        name = 'virtual_5changes'
        name_variation = name+'_'+str(variation)
        drifts = [2000, 4000, 6000, 8000]
    
    elif(number==4):
        name = 'virtual_9changes'
        name_variation = name+'_'+str(variation)
        drifts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    
    elif(number==5):
        name = 'SEA'
        name_variation = name+'_'+str(variation)
        drifts = [2000, 4000, 6000]
    
    elif(number==6):
        name = 'SEARec'
        name_variation = name+'_'+str(variation)
        drifts = [2000, 4000, 6000, 8000, 10000, 12000, 14000]
    
    print(name_variation)
    return name, name_variation, drifts
    
# options to query        

def calculateLongAccuracy(target, predict, batch):
    '''
    method to calculate the model accuracy a long time
    :param: target:
    :param: predict:
    :param: batch:
    :return: time series with the accuracy 
    '''
        
    time_series = []
    for i in range(len(target)):
        if(i % batch == 0):
            time_series.append(accuracy_score(target[i:i+batch], predict[i:i+batch]))
                               
    return time_series

def plotStreamAccuracy(models, numberDataset, variation, batch):
    '''
    method to create different types of graph for several models
    '''
    plt.style.use('seaborn-whitegrid')
    
    real_name, name, _ = chooseDataset(numberDataset, variation)

    for i in range(len(models)):
        
        # receiving the target and predict
        #predict = pd.read_excel('../projects/'+name+'-predictions.xls', sheetname=models[i])
        predict = np.asarray(pd.read_csv('../projects/new/'+models[i]+'-'+name+'.csv')['predictions'])
        target = np.asarray(pd.read_csv('../projects/new/'+models[i]+'-'+name+'.csv')['target'])
        
        # calculating the accuracy a long time
        time_series = calculateLongAccuracy(target, predict, batch)
        
        # plotting 
        text = models[i] + ": %.3f" % accuracy_score(target, predict)
        plt.plot(time_series, label=text)
    
    plt.title(real_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Batches')
    plt.legend()
    plt.show()

def plotStreamAccuracyMean(models, numberDataset, variation_max, batch):
    '''
    method to create different types of graph for several models
    '''
    plt.style.use('seaborn-whitegrid')
    
    real_name, _, _ = chooseDataset(numberDataset, variation_max)

    for model in models:
        
        # calculating the standard deviation and mean
        final_time_series = []    
        accuracy = []
        for variation in range(variation_max):
                
            _, name_variation, _ = chooseDataset(numberDataset, variation)
            # receiving the target and predict
            predict = np.asarray(pd.read_csv('../projects/new/'+model+'-'+name_variation+'.csv')['predictions'])
            target = np.asarray(pd.read_csv('../projects/new/'+model+'-'+name_variation+'.csv')['target'])
            
            # calculating the mean of accuracy
            accuracy.append(accuracy_score(target, predict))
            
            # calculating the accuracy a long time
            time_series = calculateLongAccuracy(target, predict, batch)
            
            # final time series append
            final_time_series.append(time_series)
        
        # final time series 
        time_series_mean = np.mean(final_time_series, axis=0)
        
        # plotting 
        text = model + ": %.3f (%.3f)" % (np.mean(accuracy), np.std(accuracy))
        plt.plot(time_series_mean, label=text)
    
    plt.title(real_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Batches')
    plt.legend()
    plt.show()

def plotStreamAccuracyMeanDrift(models, numberDataset, variation_max, batch):
    '''
    method to create different types of graph for several models
    '''
    plt.style.use('seaborn-white')
    
    for model in models:
        
        # calculating the standard deviation and mean
        final_time_series = []    
        accuracy = []
        for variation in range(variation_max):
                
            real_name, name_variation, drifts = chooseDataset(numberDataset, variation)
            # receiving the target and predict
            predict = np.asarray(pd.read_csv('../projects/new/'+model+'-'+name_variation+'.csv')['predictions'])
            target = np.asarray(pd.read_csv('../projects/new/'+model+'-'+name_variation+'.csv')['target'])
            
            # calculating the mean of accuracy
            accuracy.append(accuracy_score(target, predict))
            
            # calculating the accuracy a long time
            time_series = calculateLongAccuracy(target, predict, batch)
            
            # final time series append
            final_time_series.append(time_series)
        
        # final time series 
        time_series_mean = np.mean(final_time_series, axis=0)
        
        # plotting 
        text = model + ": %.3f (%.3f)" % (np.mean(accuracy), np.std(accuracy))
        plt.plot(time_series_mean, label=text)
    
    plt.title(real_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Batches')
    plt.legend()
    
    # plotting drifts
    for x, i in enumerate(drifts):
        i = i/batch
        plt.axvline(i, linestyle='dashed', color = 'black', label='Changes', alpha=0.1)
        if(x==0):
            plt.legend(loc='lower right')
    plt.show()
    
    plt.show()

def plotStreamAccuracyStandardDeviation(model, numberDataset, variation_max, batch):
    '''
    method to create different types of graph of accuracy for several models
    '''
    
    plt.style.use('seaborn-white')
    
    # calculating the standard deviation and mean
    final_time_series = []    
    accuracy = []
    for variation in range(variation_max):
            
        name, name_variation, drifts = chooseDataset(numberDataset, variation)
        # receiving the target and predict
        predict = np.asarray(pd.read_csv('../projects/new/'+model+'-'+name_variation+'.csv')['predictions'])
        target = np.asarray(pd.read_csv('../projects/new/'+model+'-'+name_variation+'.csv')['target'])
        
        # calculating the mean of accuracy
        accuracy.append(accuracy_score(target, predict))
        
        # calculating the accuracy a long time
        time_series = calculateLongAccuracy(target, predict, batch)
        
        # final time series append
        final_time_series.append(time_series)
        
    
    time_series_mean = np.mean(final_time_series, axis=0)
    time_series_std = np.std(final_time_series, axis=0)
    time_series_upper = time_series_mean + time_series_std
    time_series_lower = time_series_mean - time_series_std
        
    # plotting 
    text = "mean: %.3f (%.3f)" % (np.mean(accuracy), np.std(accuracy))
    plt.plot(time_series_upper, ':', label="upper", color='green')
    plt.plot(time_series_mean, label=text, color='red')
    plt.plot(time_series_lower, '-.', label="lower", color='blue')
    
    plt.title(model +" - dataset: "+name)
    plt.ylabel('Accuracy')
    plt.xlabel('Batches')
    
    # plotting drifts
    for x, i in enumerate(drifts):
        i = i/batch
        plt.axvline(i, linestyle='dashed', color = 'black', label='changes', alpha=0.1)
        if(x==0):
            plt.legend()
    plt.show()
    
def main():
    dataset = 6
    batch = 200
    
    models = ['Proposed Method', 'Dynse-priori', 'IGMM-CD']
    plotStreamAccuracyMeanDrift(models, dataset, 5, batch)
    
    #model = 'Proposed Method'
    #plotStreamAccuracyStandardDeviation(model, dataset, 5, batch)
    
if __name__ == "__main__":
    main()    




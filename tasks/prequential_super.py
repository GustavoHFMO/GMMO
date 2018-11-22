'''
Created on 11 de out de 2018
@author: gusta
'''

from sklearn.metrics import accuracy_score
import numpy as np


class PREQUENTIAL_SUPER():
    def __init__(self):
        '''
        Class for control the comparative algorithms
        '''
        
        self.NAME = ''
        self.TARGET = []
        self.PREDICTIONS = []
    
    def returnTarget(self):
        '''
        method to return only the target o
        '''
        
        return self.TARGET
    
    def returnPredictions(self):
        '''
        method to return only the predictions
        '''
        
        return np.asarray(self.PREDICTIONS).astype('float64')
    
    def accuracyGeneral(self):
        '''
        method to return the system accuracy for the stream
        '''
        
        y_true = self.returnTarget()
        y_pred = self.returnPredictions()
                            
        return accuracy_score(y_true, y_pred)
    
    def printIterative(self, i):
        '''
        method to show iteratively the current accuracy 
        '''
        
        current_accuracy = accuracy_score(self.TARGET, self.PREDICTIONS)*100
        percent_instances = (i*100)/len(self.STREAM)
        string = self.NAME+": %.2f -> (%d) %.2f of instances processed" % (current_accuracy, i, percent_instances)
        
        print(string)
    

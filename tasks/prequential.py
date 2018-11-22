'''
Created on 22 de ago de 2018
@author: gusta
'''

from data_streams.adjust_labels import Adjust_labels
from tasks.prequential_super import PREQUENTIAL_SUPER
al = Adjust_labels()
from streams.readers.arff_reader import ARFFReader
from detectors.ewma import EWMA
from detectors.no_detector import noDetector
from gaussian_models.gmm import GaussianMixture
from gaussian_models.kdnagmm import KDNAGMM
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class Prequential(PREQUENTIAL_SUPER):
    def __init__(self, name, labels, stream, classifier, detector, strategy, window_size, batch_size=None):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        # main variables
        self.NAME = name
        self.STREAM = al.adjustStream(labels, stream)    
        self.CLASSIFIER = classifier
        self.DETECTOR = detector
        self.STRATEGY = strategy
        self.WINDOW_SIZE = window_size
        self.BATCH_SIZE = batch_size
        self.LOSS_STREAM = []
        self.PREDICTIONS = []
        self.TARGET = []
        self.DETECTIONS = [0]
        self.CLASSIFIER_READY = True
        
        # variables to analyze the update
        self.error = []
        self.correction_error = []
    
    def run(self):
        '''
        method to run the stream
        '''
        
        # obtaining the initial window
        W = self.STREAM[:self.WINDOW_SIZE]
        
        # instantiating a window for warning levels 
        W_warning = []
        
        # instantiating the validation window
        if(self.STRATEGY):
            W_validation = W 
        
        # training the classifier
        self.CLASSIFIER = self.trainClassifier(W)
        
        # fitting the detector
        self.DETECTOR.fit(self.CLASSIFIER, W) 
        
        # plot the classifier
        #self.CLASSIFIER.plotGmm(self.CLASSIFIER, 0)
        
        # plot the classifier
        #self.CLASSIFIER.plotGmmAnalysis(self.CLASSIFIER, 0)
        
        # for to operate into a stream
        for i, X in enumerate(self.STREAM[self.WINDOW_SIZE:]):
            
            # split the current example on pattern and label
            x, y = X[0:-1], int(X[-1])
            
            # using the classifier to predict the class of current label
            yi = self.CLASSIFIER.predict(x)
            
            # storing the predictions
            self.PREDICTIONS.append(yi)
            self.TARGET.append(y)
            
            # activating the strategy
            if(self.STRATEGY):
                
                # sliding the current observation into W
                W_validation = self.slidingWindow(W_validation, X)
                
                # updating the gaussian if the classifier miss
                new_pred = self.CLASSIFIER.strategy(y, yi, x, W_validation, i)
                                    
                # storing the correction
                if(new_pred != None):
                    self.error.append(y)
                    self.correction_error.append(new_pred)
                        
            # verifying the claassifier
            if(self.CLASSIFIER_READY):
                
                # sliding the current observation into W
                W = self.slidingWindow(W, X)
            
                # monitoring the datastream
                warning_level, change_level = self.DETECTOR.detect(y, yi)
                        
                # trigger the warning strategy
                if(warning_level):
                    #print('warning level')
                        
                    # managing the window warning
                    W_warning = self.manageWindowWarning(W_warning, X)
                        
                # trigger the change strategy    
                if(change_level):
                    #print('change level')
                        
                    # storing the time of change
                    self.DETECTIONS.append(i)
                    
                    # reseting the detector
                    self.DETECTOR.reset()
                        
                    # reseting the window
                    W = self.transferKnowledgeWindow(W, W_warning)
                        
                    # reseting the classifier 
                    self.CLASSIFIER_READY = False
                    
            
            elif(self.WINDOW_SIZE > len(W)):
                
                # sliding the current observation into W
                W = self.incrementWindow(W, X)
            
            else:
                
                # training the classifier
                self.CLASSIFIER = self.trainClassifier(W)
                #self.CLASSIFIER.plotGmm(i, show=True)
                
                # fitting the detector
                self.DETECTOR.fit(self.CLASSIFIER, W) 
                        
                # plot the classifier
                #self.CLASSIFIER.plotGmm(self.CLASSIFIER, i)
                
                # releasing the new classifier
                self.CLASSIFIER_READY = True
           
            # print the current process
            self.printIterative(i)
        
    def trainClassifier(self, W):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''
        
        # split patterns and labels
        x, y = W[:,0:-1], W[:,-1]
        
        X_train, y_train = x, y
        
        '''
        # breaking the dataset non-overlapping on 75% training and 25% to test 
        skf = StratifiedKFold(y, n_folds=4)
        # taking the current index for training and test
        train_index, test_index = next(iter(skf))
        
        # obtendo os conjuntos de dados
        X_train = x[train_index]
        y_train = y[train_index]
        X_test = x[test_index]
        y_test = y[test_index]
        '''
        
        # fitting the dataset
        self.CLASSIFIER.fit(X_train, y_train)
        # accuracy for train
        #print("accuracy train: ", accuracy_score(y_train, self.CLASSIFIER.predict(X_train)))
        
        return self.CLASSIFIER
    
    def patternsPerClass(self, W):
        '''
        method to divide patterns per class
        :param: W: window with all patterns
        '''
        
        unique, _ = np.unique(W[:,-1], return_counts=True)
        classes = len(unique)
        
        patterns = np.asarray([None] * classes) 
        for j in range(classes):
            patterns[j] = np.asarray([W[i] for i in range(len(W[:, -1])) if W[i,-1]==j])
        
        return patterns
        
    def slidingWindow(self, W, x):
        '''
        method to slide the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * len(W)
        aux[0:-1] = W[1:]
        aux[-1] = x
    
        return np.asarray(aux)

    def incrementWindow(self, W, x):
        '''
        method to icrement the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * (len(W)+1)
        aux[:-1] = W
        aux[-1] = x
        
        return np.asarray(aux) 
        
    def resetWindow(self, W):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        return np.array([])
    
    def manageWindowWarning(self, W, x):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        if(self.CLASSIFIER_READY):
            W = self.incrementWindow(W, x)
            
            if(len(W) > self.WINDOW_SIZE/2):
                W = self.resetWindow(W)
        
        return W
        
    def transferKnowledgeWindow(self, W, W_warning):    
        '''
        method to transfer the patterns of one windown to other
        :param: W: window that will be updated
        :param: W_warning: window with the patterns that will be passed
        '''
        
        W = W_warning
        
        return W
    
    def percentAccuracyCorrection(self):
        '''
        method to return the percentage of adjustment made a long time 
        '''
        
        accuracy = (len(self.correction_error)*100)/len(self.STREAM[self.WINDOW_SIZE:])
        
        return accuracy, len(self.correction_error)
    
    def accuracyAfterCorrection(self):
        '''
        method to return the system accuracy for the adjustment of wrong observations
        '''
        
        accuracy = accuracy_score(self.error, self.correction_error)
        
        
        return accuracy
    
def main():
    
    i = 3
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes']
    
    #1. import the stream
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+"/"+dataset[i]+"_"+str(0)+".arff")
    #labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/virtual_5changes.arff")
    #labels, _, stream_records = ARFFReader.read("../data_streams/Dynse/SEA.arff")

    #2. import the classifier
    classifier = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=True)
    
    #3. import the detector
    #nodetector = noDetector()
    m = 50
    detector = EWMA(min_instance=m, lt=1)
    #detector = noDetector()
    
    #4. instantiate the prequetial
    preq = Prequential(name="KDNAGMMO+ECDD",
                       labels=labels,
                       stream=stream_records,
                       classifier=classifier,
                       detector=detector,
                       strategy=True,
                       window_size=m)
    
    #5. execute the prequential
    preq.run()
    
    # storing only the predictions
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+".csv")
    
if __name__ == "__main__":
    main()        
           
        
    
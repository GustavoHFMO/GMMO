'''
Created on 22 de ago de 2018
@author: gusta
'''

from streams.readers.arff_reader import ARFFReader
from detectors.ewma import EWMA
from detectors.no_detector import noDetector
from classifiers.gmm import GaussianMixture
from classifiers.agmm import AGMM
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

class Prequential():
    def __init__(self, name, stream, classifier, detector, strategy, window_size, batch_size):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        def adjustStream(stream):
            '''
            method to replace strings on stream per int
            :param: stream: current stream
            '''
            for i in range(len(stream)):
                if(stream[i][2]=='n'):
                    stream[i][2]=0
                elif(stream[i][2]=='p'):
                    stream[i][2]=1
                elif(stream[i][2]=='3'):
                    stream[i][2]=0
                elif(stream[i][2]=='2'):
                    stream[i][2]=2
                elif(stream[i][2]=='1'):
                    stream[i][2]=1
                    
            return np.asarray(stream)
                    
            return np.asarray(stream)

        # main variables
        self.NAME = name
        self.STREAM = adjustStream(stream)    
        self.CLASSIFIER = classifier
        self.DETECTOR = detector
        self.STRATEGY = strategy
        self.WINDOW_SIZE = window_size
        self.BATCH_SIZE = batch_size
        self.LOSS_STREAM = []
        self.PREDICTIONS = []
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
        
        # defining windows that will store the last predictions
        batch_pred = [None] * self.BATCH_SIZE
        batch_true = [None] * self.BATCH_SIZE 
        
        # for to operate into a stream
        for i, X in enumerate(self.STREAM[self.WINDOW_SIZE:]):
            
            # split the current example on pattern and label
            x, y = X[0:-1], int(X[-1])
            
            # using the classifier to predict the class of current label
            yi = self.CLASSIFIER.predict(x)
            
            # storing the predictions
            self.PREDICTIONS.append(yi)
            
            # storing the prediction and true label
            batch_pred = self.slidingWindow(batch_pred, yi)
            batch_true = self.slidingWindow(batch_true, y)
            
            # computing the loss to the current batch
            if(i != 0 and i % self.BATCH_SIZE == 0):
                loss = accuracy_score(batch_true, batch_pred)
                self.LOSS_STREAM.append(loss)
            
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
                    print('warning level')
                        
                    # managing the window warning
                    W_warning = self.manageWindowWarning(W_warning, X)
                        
                # trigger the change strategy    
                if(change_level):
                    print('change level')
                        
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
                        
                # fitting the detector
                self.DETECTOR.fit(self.CLASSIFIER, W) 
                        
                # plot the classifier
                #self.CLASSIFIER.plotGmm(self.CLASSIFIER, i)
                
                # releasing the new classifier
                self.CLASSIFIER_READY = True
           
            # print the current process
            print(self.NAME+": "+str((i*100)/len(self.STREAM))+"% of instances are prequentially processed!")
        
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
        print("accuracy train: ", accuracy_score(y_train, self.CLASSIFIER.predict(X_train)))
        
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
    
    def returnTarget(self):
        '''
        method to return only the target o
        '''
        
        return self.STREAM[self.WINDOW_SIZE:,-1]
    
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
        
    def plot(self):
        '''
        method to plot the acuracy of the classifier on the stream
        '''
        
        plt.plot(self.LOSS_STREAM)
        plt.title("Dataset: sine")
        plt.ylabel('acuracy score')
        plt.xlabel('batch size: '+str(self.BATCH_SIZE))
        plt.show()
            
def main():
    
    #1. import the stream
    _, _, stream_records = ARFFReader.read("data_streams/_synthetic/virtual.arff")
    #_, _, stream_records = ARFFReader.read("data_streams/_synthetic/circles.arff")
    #_, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine1.arff")
    #_, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine2.arff")
    
    m = 300
    batch = 250
    lt = 1
    
    ################################################################################## 1 #######################################################
    #2. import the classifier
    #gmm = AGMM()
    gmm = GaussianMixture()
    
    #3. instantiate the detector
    #ewma = EWMA(min_instance=m, lt=lt)
    nodetect = noDetector()
    
    #4. instantiate the prequetial
    prequential = Prequential(name="",
                              stream=stream_records,
                              classifier=gmm,
                              detector=nodetect,
                              strategy=True,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    prequential.run()
    ############################################################################################################################################
    
    ################################################################################## 2 #######################################################
    #2. import the classifier
    #gmm = AGMM()
    gmm = GaussianMixture()
    
    #3. instantiate the detector
    nodetect = noDetector()
    
    #4. instantiate the prequetial
    prequential1 = Prequential(name="",
                               stream=stream_records,
                              classifier=gmm,
                              detector=nodetect,
                              strategy=True,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    prequential1.run()
    ############################################################################################################################################
    
    ################################################################################## 3 #######################################################
    #2. import the classifier
    #gmm = AGMM()
    gmm = GaussianMixture()
    
    #3. instantiate the detector
    ewma = EWMA(min_instance=m, lt=lt)
    
    #4. instantiate the prequetial
    prequential2 = Prequential(name="",
                               stream=stream_records,
                              classifier=gmm,
                              detector=ewma,
                              strategy=False,
                              window_size=m,
                              batch_size=batch)
    
    #5. execute the prequential
    prequential2.run()
    ############################################################################################################################################
    
    
    # printing the results
    #plt.plot(prequential.LOSS_STREAM, label = "AGMMO: %f" % prequential.accuracyGeneral())
    print("AGMMO: %4.f" % prequential.accuracyGeneral())
    print("AGMMO: correction: ", prequential.percentAccuracyCorrection()[1], 
          " - ", prequential.percentAccuracyCorrection()[0], "% of instances ", 
          "accuracy correction: ", prequential.accuracyAfterCorrection())
    
    #plt.plot(prequential1.LOSS_STREAM, label = "AGMMO+ECDD: %f" % prequential1.accuracyGeneral())
    print("AGMMO+ECDD: %4.f" % prequential1.accuracyGeneral())
    print("AGMMO+ECDD: after correction: ", prequential1.percentAccuracyCorrection()[1], 
          " - ", prequential1.percentAccuracyCorrection()[0], "% of instances ", 
          "accuracy correction: ", prequential1.accuracyAfterCorrection())
    
    plt.ylabel('Accuracy')
    plt.xlabel('Batches')
    plt.legend()
    plt.show()
    
    '''
    #2. import the classifier
    gmm = AGMM()
    #gmm = GaussianMixture()
    
    #3. import the detector
    #nodetector = noDetector()
    m = 300
    #ewma = EWMA(min_instance=m, lt=1)
    ewma = noDetector()
    
    #4. instantiate the prequetial
    prequential = Prequential(name="",
                              stream=stream_records,
                              classifier=gmm,
                              detector=ewma,
                              strategy=True,
                              window_size=m,
                              batch_size=20)
    
    #5. execute the prequential
    prequential.run()
    
    print(prequential.accuracyGeneral())
    print("AGMMO: after correction: ", prequential.percentAccuracyCorrection(), "% of instances ", 
          "accuracy correction: ", prequential.accuracyAfterCorrection())
    
    #6. plot the execution
    prequential.plot()
    '''
    
    
if __name__ == "__main__":
    main()        
           
        
    
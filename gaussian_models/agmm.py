'''
Created on 29 de abr de 2018
@author: gusta
'''

from sklearn.model_selection import train_test_split
from data_streams.static_datasets import Datasets
from gaussian_models.gmm_super import GMM_SUPER
from gaussian_models.gmm import GaussianMixture
import numpy as np

class AGMM(GMM_SUPER):
    def __init__(self):
        self.NAME = 'AGMM'
    
    def fit(self, train_input, train_target, type_selection='AIC', Kmax=4, restarts=1, iterations=25):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: type_selection: name of prototype selection metric. Default 'AIC'
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # storing the patterns
        self.train_input = train_input
        self.train_target = train_target

        # receiving the number of classes
        unique, _ = np.unique(train_target, return_counts=True)
        self.L = len(unique)
        
        # dividing the patterns per class
        classes = []
        for i in range(self.L):
            aux = []
            for j in range(len(self.train_target)):
                if(self.train_target[j] == i):
                    aux.append(self.train_input[j])
            classes.append(np.asarray(aux))
            
        # variable to store the weight of each gaussian
        self.dens = []
        
        # instantiating the gaussians the will be used
        self.gaussians = []
         
        # creating the optimal gaussians for each class
        for i in range(len(classes)):
            # EM with BIC applied for each class
            gmm = self.chooseBestModel(classes[i], type_selection, Kmax, restarts, iterations)

            # storing the gaussians            
            for gaussian in gmm.gaussians:
                gaussian.label = i 
                self.gaussians.append(gaussian)
                
            # storing the density of each gaussian
            for k in gmm.dens:
                self.dens.append(k)    
        
        # defining the number of gaussians for the problem
        self.K = len(self.gaussians)
        
        # defining the weight of each gaussian based on dataset
        self.mix = [self.dens[i]/len(self.train_target) for i in range(self.K)]
     
        # intializing the matrix of weights
        self.matrixWeights = self.Estep()
        
    def predict_one(self, x):
        '''
        method to predict the class for a only pattern x
        :param: x: pattern
        :return: the respective label for x
        '''
        
        # receiving the gaussian with more probability for the pattern
        y = [0]*self.K
        for i in range(self.K):
            y[i] = self.posteriorProbability(x, i)
        gaussian = np.argmax(y)    
                    
        # returning the label
        return self.gaussians[gaussian].label
    
    def probs_reference(self, y, x):
        '''
        method to return the probs of an example for each gaussian
        :param: references: of gaussians 
        :param: x: pattern 
        :return: the probability for each gaussian
        '''

        # computing the gmm density
        z = []
        for i in range(len(self.gaussians)):
            if(self.gaussians[i].label == y):
                z.append(self.conditionalProbability(x, i))
        dens = np.sum(z)
        
        # receiving the gaussian with more probability for the pattern
        z = [0] * len(self.gaussians)
        for i in range(len(self.gaussians)):
            if(self.gaussians[i].label == y):
                z[i] = (self.conditionalProbability(x, i))/dens
            
        return np.argmax(z)
    
    def updateWeights(self, data):
        '''
        method to update the weights of each gaussian
        '''        
        
        # updating the relevance of each gaussian
        matrixWeights = self.Estep()
        
        # variable to store the new mix
        new_mix = [0] * self.K
        
        # updating the weight of each gaussian
        for i in range(self.K):
            
            # dividing the probabilities to each cluster
            wgts = matrixWeights[:,i]
            # this variable is going to responsible to storage the sum of probabilities for each cluster
            dens = np.sum(wgts)
            # compute new mix
            new_mix[i] = dens/len(self.train_input)
        
        return new_mix    
        
    def strategy(self, y_true, y_pred, x, W, i):
        '''
        method to update an gaussian based on error 
        '''
        
        # verify classification
        if(y_true != y_pred):
            
            # adjusting the window
            W = np.asarray(W)
                    
            # updating the data for train and target
            self.train_input = W[:,0:-1]
            self.train_target = W[:,-1]
            
            # receiving the gaussian for the true class which is more near for the instance x
            gaussian = self.probs_reference(y_true, x)
            
            # get data from the class that was classified wrong
            data = W[W[:,-1] == y_true]
            
            # storing the observations which belong to the responsible gaussian
            data_update = []
            for j in data[:,0:-1]:
                z = self.probs_reference(y_true, j)
                if(z == gaussian):
                    data_update.append(j)
            data_update = np.asarray(data_update)
            
            # updating the parameters of gaussian
            self.gaussians[gaussian].mu = np.mean(data_update, axis = 0)
            self.gaussians[gaussian].sigma = np.cov(np.transpose(data_update))
            
            # updating the gaussians weights
            self.mix = self.updateWeights(self.train_input)
            
            # returning the new prediction for later analysis 
            return self.predict(x)
            
        return None

def main():
   
    dt = Datasets()
    X, y = dt.chooseDataset(7)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    ########################## gaussian without modification ####################################
    # instantiating the gaussian
    gmm = GaussianMixture()
    
    # fitting the dataset
    gmm.fit(X_train, y_train)
    
    # making predictions
    predictions = gmm.predict(X_train)
    # evaluating the accuracy for the training dataset
    train_accuracy = np.mean(predictions == y_train) * 100
    print('train accuracy: %.1f' % train_accuracy)
    
    # making predictions
    predictions = gmm.predict(X_test)
    # evaluating the accuracy for the test dataset
    test_accuracy = np.mean(predictions == y_test) * 100
    print('test accuracy: %.1f' % test_accuracy)
    
    # plotting the gmm created
    gmm.plotGmmTrainTest(gmm, train_accuracy, test_accuracy)
    #gmm.plotGmm(gmm, 0)
    ############################################################################################
    
    ########################## gaussian with modification ####################################
    # instantiating the gaussian
    gmm = AGMM()
    
    #gmm.fitClustering(X_train, 5)
    #gmm.animation(30)
    
    # fitting the dataset
    gmm.fit(X_train, y_train)
    
    # making predictions
    predictions = gmm.predict(X_train)
    # evaluating the accuracy for the training dataset
    train_accuracy = np.mean(predictions == y_train) * 100
    print('train accuracy: %.1f' % train_accuracy)
    
    # making predictions
    predictions = gmm.predict(X_test)
    # evaluating the accuracy for the test dataset
    test_accuracy = np.mean(predictions == y_test) * 100
    print('test accuracy: %.1f' % test_accuracy)
    
    # plotting the gmm created
    gmm.plotGmmTrainTest(gmm, train_accuracy, test_accuracy)
    ############################################################################################
    
if __name__ == "__main__":
    main()        
            
            
            
            
        
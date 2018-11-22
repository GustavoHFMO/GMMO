'''
Created on 29 de abr de 2018
@author: gusta
'''

from data_streams.static_datasets import Datasets
from gaussian_models.gmm_super import GMM_SUPER
from gaussian_models.gaussian import Gaussian
import numpy as np

class GaussianMixture(GMM_SUPER):
    def __init__(self):
        self.NAME = 'GMM'

    def fit(self, train_input, train_target):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        '''
        
        # storing the patterns
        self.train_input = train_input
        self.train_target = train_target
        
        # storing the number of classes 
        unique, counts = np.unique(train_target, return_counts=True)
        self.K = len(unique)
        self.L = self.K
        
        # storing the mix
        self.mix = [counts[i]/len(self.train_target) for i in range(self.K)] 
                
        # computing the mean and covariance for each cluster
        self.gaussians = []
        for i in range(self.K):
            # taking the respectively data for each class
            g = Gaussian(np.mean(train_input[train_target == i], axis=0), np.cov(np.transpose(train_input[train_target == i])), i)
            # storing the gaussians
            self.gaussians.append(g)

        # auxiliary variable to plot the dataset
        self.ismatrix = True
                
        # intializing the matrix of weights
        self.matrixWeights = self.Estep()
   
    def predict_one(self, x):
        '''
        method to predict the class for a pattern x
        :param: x: pattern
        :return: the respective label for x
        '''
        
        # calculating the density
        y = [0]*self.K
        for i in range(self.K):
            y[i] = self.posteriorProbability(x, i)
            
        label = np.argmax(y)
        return label
    
    def strategy(self, y_true, y_pred, x, W, i):
        '''
        method to update an gaussian based on error 
        '''
        
        # adjusting the window
        W = np.asarray(W)
        
        # verify classification
        if(y_true != y_pred):
            
            '''
            # printing the previous result
            print("correct class:             ", y_true)
            print("class predicted:           ", y_pred)
                
            # printing the new observation
            markers = ["^", "o"]
            plt.scatter(x[0], x[1], color = 'green', marker=markers[int(y_true)])
                
            # plot the respective label of the new observation
            #pbs = self.probs(x)
            #string = "[%d][%d][%.2f, %.2f]" % (i, y_pred, pbs[0], pbs[1])
            #plt.text(x[0], x[1], string)
              
            # plotting the result of each instance of validation window
            #self.plotGmmAnalysis(self, i)
            self.plotGmm(self, i)
            '''
            
            # updating the gaussian which missed the new observation
            data = W[W[:,-1] == y_true]
            self.gaussians[y_true].mu = np.mean(data[:,0:-1], axis = 0)
            self.gaussians[y_true].sigma = np.cov(np.transpose(data[:,0:-1]))
                
            # storing the number of classes 
            unique, counts = np.unique(W[:,-1], return_counts=True)
            self.K = len(unique)
                
            # updating the mix of each gaussian
            self.mix = [counts[i]/len(W[:,-1]) for i in range(self.K)]

            '''
            # printing the new observation after the update
            print("correct class:             ", y_true)
            print("after correction:          ", self.predict(x))
            plt.scatter(x[0], x[1], color = 'green', marker=markers[int(y_true)])
                
            # plot the respective label of the new observation after the update
            #pred = self.predict(x)
            #pbs = self.probs(x)
            #string = "[%d][%d][%.2f, %.2f]" % (i, pred, pbs[0], pbs[1])
            #plt.text(x[0], x[1], string)
                             
            # plotting the result of each instance of validation window
            #self.plotGmmAnalysis(self, i)
            self.plotGmm(self, i)
            '''
            
            # returning the new prediction for later analysis 
            return self.predict(x)
        
        return None
    
def main():
    print()
    
if __name__ == "__main__":
    main()        
            
            
            
            
        
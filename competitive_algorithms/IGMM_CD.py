'''
Created on 17 de set de 2018
@author: gusta
'''

from data_streams.adjust_labels import Adjust_labels
from tasks.prequential_super import PREQUENTIAL_SUPER
al = Adjust_labels()
from streams.readers.arff_reader import ARFFReader
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import array
import pandas as pd
import numpy as np
plt.style.use('seaborn-whitegrid')

class Gaussian:
    def __init__(self, mu, sigma, reference):
        '''
        Constructor of the Gaussian distribution
        :param: mu: the average of the data
        :param: sigma: the standard deviation of the data
        '''
        self.mu = mu
        self.sigma = sigma
        self.reference = reference
        
    def pdf_scalar(self, x):
        '''
        Method to compute the probability of an scalar
        :param: x: variable x which will be computed
        :return: the probability of the variable x belongs to this gaussian distribution
        '''
        
        u = (x-self.mu)/np.abs(self.sigma)
        y = (1/(np.sqrt(2*np.pi) * np.abs(self.sigma))) * np.exp(-u*u/2)
        return y 
    
    def pdf_vector(self, x):
        '''
        Method to compute the probability of an vector
        :param: x: variable x which will be computed
        :return: the probability of the variable x belongs to this gaussian distribution
        '''
        
        x = np.transpose(array([x]))
        mu = np.transpose(array([self.mu]))
        
        # dealing with possible divisions by zero
        part1 = 1/(np.power(2*np.pi, len(x)/2) * np.sqrt(np.linalg.det(self.sigma)))
        if(part1 == np.float('inf')): part1 = 0
        
        part2 = np.transpose(np.subtract(x, mu))
        
        # dealing with zero on matrix
        try:
            part3 = np.linalg.inv(self.sigma)
        except:
            part3 = np.linalg.pinv(self.sigma)
        
        part4 = np.subtract(x, mu)
        
        # calculation of matrices
        a = np.dot(part2, part3)
        b = np.dot(a, part4)
        b = -0.5 * b[0][0]
        
        # an way to avoid problems with large values in b, because it result on ifinity results
        c = np.exp(b)
        
        y = part1 * c
        
        return y 
        
    def printstats(self):
        '''
        method to print the current mu and sigma of the distribution
        '''
        print('Gaussian: mi = {:.2}, sigma = {:.2}'.format(self.mu, self.sigma))
        
class IGMMCD(PREQUENTIAL_SUPER):
    def __init__(self, sigma_ini, cver, T):
        # starting the parameters
        # prior probability
        self.sigma_ini = sigma_ini
        self.cver = cver
        self.T = T
        self.train_input = []
        self.train_target = []
        self.gaussians = []  
        self.mix = []
        self.sp = []
        self.NAME = 'IGMM-CD'
        
        # variable to count the cross validation
        self.count = 0

    def fit(self, x, y):
        '''
        method to create the first gaussian
        :param: x: the example that will be clusterized
        '''
        
        # mean 
        mu_ini = x
        
        # covariance
        cov_ini = (self.sigma_ini**2) * np.identity(len(x))

        # prior probability
        self.mix.append(1)
        
        # starting the max value of likelihood
        self.sp.append(1)
        
        # starting the first gaussian
        self.gaussians.append(Gaussian(mu=mu_ini, sigma=cov_ini, reference=y))
        
        # updating the other components
        self.updateLikelihood(x)
        self.updateWeight()
    
    def predict_one(self, x):
        '''
        method to predict the class for only one pattern x
        :param: x: pattern
        :return: the respective label for x
        '''
        
        y = [0]*len(self.gaussians)
        for i in range(len(self.gaussians)):
            y[i] = self.conditionalProbability(x, i)
        label = self.gaussians[np.argmax(y)].reference
        return label
    
    def predict(self, x):
        '''
        method to predict the class for a several patterns x
        :param: x: pattern
        :return: the respective label for x
        '''

        if(len(x.shape) > 1):
            labels = []
            for pattern in x:
                labels.append(self.predict_one(pattern))
            return labels
        
        else:
            return self.predict_one(x)
        
    def posteriorProbability(self, x, i):
        '''
        method to return the posterior probability of an variable x to a gaussian i
        :param: x: observation
        :param: i: number of the gaussian
        '''
    
        dens = []
        for j in range(len(self.gaussians)):
            dens.append(self.conditionalProbability(x, j))
        
        # to avoid nan
        dens = np.nan_to_num(dens) 
        dens = np.sum(dens) 
        if(dens == 0.0): dens = 0.01  
        
        posterior = (self.conditionalProbability(x, i))/dens
        
        return posterior
        
    def conditionalProbability(self, x, i):
        '''
        method to return the conditional probability of an variable x to a gaussian i
        :param: x: observation
        :param: i: number of the gaussian
        '''

        return self.gaussians[i].pdf_vector(x)*self.mix[i]
    
    def updateLikelihood(self, x):
        '''
        method to update the parameter cver
        :param: x: new observation
        '''
        
        # updating the loglikelihood
        for i in range(len(self.gaussians)):
            self.sp[i] = self.sp[i] + self.posteriorProbability(x, i)
        
    def updateMean(self, x, i):
        '''
        Method to update the mean of a gaussian i
        return new mean
        '''
        
        part1 = self.posteriorProbability(x, i)/self.sp[i]
        part2 = np.subtract(x, self.gaussians[i].mu)
        new = self.gaussians[i].mu + (np.dot(part1, part2))
        
        return new
        
    def updateCovariance(self, x, i, old_mean):
        '''
        Method to update the covariance of a gaussian i
        return new covariance
        '''
        part0 = self.gaussians[i].sigma
        part1 = np.subtract(self.gaussians[i].mu, old_mean)
        part2 = np.transpose(part1)
        part3 = np.dot(part1, part2)
        part4 = np.subtract(part0, part3)

        #plus
        part5 = self.posteriorProbability(x, i)/self.sp[i]
        part6 = np.subtract(x, self.gaussians[i].mu)
        part7 = np.transpose(part6)
        part8 = np.dot(part6, part7)
        part9 = np.subtract(part8, part0)
        part10 = np.dot(part5, part9)
        
        #final
        covariance = np.add(part4, part10) 
        
        return covariance
    
    def updateWeight(self):
        '''
        Method to update the mix
        '''
        
        dens = np.sum(self.sp)
        if(dens == 0.0): dens = 0.01
        
        for i in range(len(self.gaussians)):
            self.mix[i] = self.sp[i]/dens
            
    def verifyComponents(self, x, y_true, y_pred):
        '''
        method to verify if there are any component that represent the variable x
        :param: x: observation
        :param: y: label    
        '''
        
        new_component = True
        for i in range(len(self.gaussians)):
            
            if(self.gaussians[i].reference == y_true):
                
                prob = self.conditionalProbability(x, i)
                part0 = (len(x)/2)
                part1 = (2*np.pi)** part0
                part2 = np.sqrt(np.linalg.det(self.gaussians[i].sigma))
                crit = self.cver/(part1*part2)
                    
                if(prob >= crit):
                    new_component = False
                
        if(new_component == True):
            self.fit(x, y_true)
        else:
            self.updateComponents(x, y_true)
            
    def updateComponents(self, x, y_true):
        '''
        method to update the current gaussians
        '''
        
        # storing the conditional probability to the gaussians that belongs to the current observation class
        probs = []
        for i, gaussian in enumerate(self.gaussians):
            if(y_true == gaussian.reference):
                probs.append(self.conditionalProbability(x, i))
            else:
                probs.append(0)

        # receiving the owner of observation 
        gaussian = np.argmax(probs)

        # updating weight and sp for all gaussians
        self.updateLikelihood(x)
        
        # updating the parameters of the observation owner        
        old_mean = self.gaussians[gaussian].mu
        self.gaussians[gaussian].mu = self.updateMean(x, gaussian)
        self.gaussians[gaussian].sigma = self.updateCovariance(x, gaussian, old_mean)
        
        # updating weight and sp for all gaussians
        self.updateWeight()
        
    def removeComponents(self):
        '''
        method to remove components
        '''
        
        # receiving the classes
        classes = np.unique([i.reference for i in self.gaussians])
        
        # storing the indexes of each gaussian per class
        references = []
        for i in classes:
            aux = []
            for j, gaussians in enumerate(self.gaussians):
                if(gaussians.reference == i):
                    aux.append(j) 
            references.append(aux)
            
        reset = False
        # removing the gaussians
        for gaussians in references:
            if(len(gaussians)>self.T):
                probs = []
                refs = []
                for j in gaussians:
                    probs.append(self.mix[j])
                    refs.append(j)
                min_ = refs[np.argmin(probs)]
                del self.gaussians[min_]
                del self.mix[min_]
                del self.sp[min_] 
                
                reset = True
                
                
        # reseting the parameter sp
        if(reset==True):
            self.resetSP()
        
        # updating the weight and sp for all gaussians
        self.updateWeight()
    
    def resetSP(self):
        '''
        method to reset the parameter sp
        '''
        
        for i in range(len(self.gaussians)):
            self.sp[i] = 1
    
    def plotGmm(self, bestGMM, t):
        
        # creating the image
        plt.subplot(111)
        
        # classes
        unique = np.unique(bestGMM.train_target)
        
        # defining the colors of each gaussian
        colors = cm.rainbow(np.linspace(0, 1, 4))
        marks = ["^", "o", '+', ',']

        # receiving each observation per class
        train_input = np.asarray(bestGMM.train_input)
        classes = []
        for i in unique:
            aux = []
            for j in range(len(bestGMM.train_target)):
                if(bestGMM.train_target[j] == i):
                    aux.append(train_input[j])
            classes.append(np.asarray(aux))
                
            
        # plotting each class
        classes = np.asarray(classes)
        for i in range(len(unique)):
            plt.scatter(classes[i][:,0],
                        classes[i][:,1],
                        color = colors[i],
                        marker = marks[i],
                        label = 'class '+str(i)) 
            
            
        # plotting the gaussians under the dataset
        for i in range(len(bestGMM.gaussians)):
            self.draw_ellipse(bestGMM.gaussians[i].mu, 
                              bestGMM.gaussians[i].sigma, 
                              colors[int(bestGMM.gaussians[i].reference)])
            
        # definindo o titulo e mostrando a imagem
        plt.title('GMM - time: ' +str(t))
        plt.legend()
        plt.show()

    def draw_ellipse(self, position, covariance, color, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, _ = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        # Draw the Ellipse
        for nsig in range(2, 3):
            ax.add_patch(patches.Ellipse(position, 
                                         nsig * width, 
                                         nsig * height,
                                         angle, 
                                         #fill=False,
                                         color = color,
                                         alpha=0.3, 
                                         **kwargs))
       
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

    def prequential(self, labels, stream, window_size):
        '''
        method to run the IGMM-CD on a specific stream
        :param: labels: existing labels on datastream
        :param: stream: data that will be runned
        '''
        
        self.STREAM = al.adjustStream(labels, stream)
        
        # first observation
        x, y = self.STREAM[0][:-1], self.STREAM[0][-1]
    
        # starting the model    
        self.fit(x, y)
        
        # to plot 
        #self.train_input.append(x)
        #self.train_target.append(y)
        #self.plotGmm(self, 0)
        
        # variable to store the predictions and error
        self.PREDICTIONS = []
        self.TARGET = []
        
        # data stream
        for i, X in enumerate(self.STREAM):
            
            # receiving the patterns and the respective label
            x, y = X[0:-1], int(X[-1])
            
            # predicting the class of the observation
            y_pred = self.predict(x) 

            # storing the predictions            
            if(i >= window_size):
                self.PREDICTIONS.append(y_pred)
                self.TARGET.append(y)
                # print the current process
                self.printIterative(i)
                            
            # verifying the components
            self.verifyComponents(x, y, y_pred)
            
            # removing the components            
            self.removeComponents()
            
    def prequential_cross_validation(self, labels, stream, window_size, fold, qtd_folds):
        '''
        method to run the IGMM-CD on a specific stream
        :param: labels: existing labels on datastream
        :param: stream: data that will be runned
        '''
        
        self.STREAM = al.adjustStream(labels, stream)
        
        # first observation
        x, y = self.STREAM[0][:-1], self.STREAM[0][-1]
    
        # starting the model    
        self.fit(x, y)
        
        # variable to store the predictions and error
        self.PREDICTIONS = []
        self.TARGET = []
        
        # data stream
        for i, X in enumerate(self.STREAM):
            
            # to use the cross validation
            if(self.cross_validation(i, fold, qtd_folds)):
                
                # receiving the patterns and the respective label
                x, y = X[0:-1], int(X[-1])
                
                # predicting the class of the observation
                y_pred = self.predict(x) 
    
                # storing the predictions            
                if(i >= window_size):
                    self.PREDICTIONS.append(y_pred)
                    self.TARGET.append(y)
                    # print the current process
                    self.printIterative(i)
                                
                # verifying the components
                self.verifyComponents(x, y, y_pred)
                
                # removing the components            
                self.removeComponents()
    
    def cross_validation(self, i, fold, qtd_folds):
        '''
        Method to use the cross validation to data streams
        '''
        
        # if the current point reach the maximum, then is reseted 
        if(self.count == qtd_folds):
            self.count = 0
            
        # false if the fold is equal to count
        if(self.count == fold):
            Flag = False
        else:
            Flag = True
        
        # each iteration is accumuled an point
        self.count += 1
        
        #returning the flag
        return Flag
        
def main():
    
    #===========================================================================
    # #1. import the stream
    # i = 0
    # dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes']
    # labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+"/"+dataset[i]+"_"+str(0)+".arff")
    #===========================================================================
    
    i = 3
    dataset = ['powersupply', 'PAKDD', 'elec', 'noaa']
    labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    stream_records = stream_records[:500]
    
    igmmcd = IGMMCD(10, 0.01, 9)
    igmmcd.prequential(labels, stream_records, 50)
    
    # printing the final accuracy
    print(igmmcd.accuracyGeneral())
    
    #===========================================================================
    # # storing only the predictions
    # df = pd.DataFrame(data={'predictions': igmmcd.PREDICTIONS})
    # df.to_csv("../projects/"+igmmcd.NAME+"-"+dataset[i]+".csv")
    #===========================================================================
    
if __name__ == "__main__":
    main()        

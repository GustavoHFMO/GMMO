'''
Created on 11 de out de 2018
@author: gusta
'''

import numpy as np

class Gaussian:
    def __init__(self, mu, sigma, label=None):
        '''
        Constructor of the Gaussian distribution
        :param: mu: the average of the data
        :param: sigma: the standard deviation of the data
        '''
        self.mu = mu
        self.sigma = sigma
        self.label = label
        
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
        
        # to avoid problems
        x = [0.01 if i == 0 else i for i in x]
        
        # adjusting the pattern
        x = np.transpose(np.array([x]))
        mu = np.transpose(np.array([self.mu]))
        
        # dealing with possible divisions by zero
        part1 = 1/(np.power(2*np.pi, len(x)/2) * np.sqrt(np.linalg.det(self.sigma)))
        if(part1 == np.float('inf')): part1 = 0.01
        
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
    

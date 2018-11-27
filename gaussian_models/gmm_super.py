'''
Created on 29 de abr de 2018
@author: gusta
'''

from gaussian_models.prototype_selection import PrototypeSelection
ps = PrototypeSelection()
from gaussian_models.gaussian import Gaussian
import matplotlib.patches as patches
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import copy
import warnings
warnings.simplefilter("ignore")

plt.style.use('seaborn-whitegrid')

class GMM_SUPER:
    def __init__(self):
        pass

    def fitClustering(self, train_input, K):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: K: integer - the quantity of Gaussians used
        '''
        
        # storing the dataset
        self.train_input = train_input
            
        # number of gaussians
        self.K = K
    
        # divinding the weights of each gaussians in relation of the total probability
        self.mix = [1./K] * K
            
        # dividing the number of examples for each gaussian
        self.N = int(np.round(0.3 * len(train_input)))
                         
        # creating the gaussians
        self.gaussians = []
            
        # allocating itens for each gaussian
        for _ in range(K):
            # extracting random examples to calculate the first parameters
            if(len(self.train_input) > 5):
                randomData = [self.train_input[np.random.randint(1, len(self.train_input)-1)] for _ in range(self.N)]
            else: 
                randomData = self.train_input
    
            # creating the variables randomly
            g = Gaussian(np.mean(randomData, axis=0), np.cov(np.transpose(randomData)))
                
            # storing the gaussians
            self.gaussians.append(g)
                
        # defining the quantity of parameters of model
        self.n = len(self.train_input)
        print(self.train_input)
        print(self.n)
        # defining the dimension of problem
        d = len(self.train_input[0])
        
        # quantity of means, covariances and constants of mixtures
        self.p = (self.K * d) + (self.K * d * d) + self.K
            
        # intializing the matrix of weights
        self.matrixWeights = self.Estep()
    
    def trainEM(self, iterations, log=None):
        '''
        method to train the gaussians
        :param: iterations: integer - quantity of iterations necessary to train the models
        :param: log: boolean - variable to show the log of train
        '''
        
        # process to train the gmm
        for i in range(iterations):
                
            # printing the process
            if(log == True):
                #self.printstats()
                print('[', i, ']:', self.loglike)

            # EM process
            self.Mstep(self.matrixWeights)
            self.matrixWeights = self.Estep()
    
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
    
    def Estep(self):
        '''
        Method to compute the estimation of probability for each data
        :return: a matrix containing the weights for each data for all clusters 
        '''
        # (p=1)
        self.loglike = 0
        
        # matrix that will storage the weights
        matrixWeights = []
        
        # for to iterate for each data point
        for x in self.train_input:
            # for to estimate each weight of gaussians
            weights = [0] * self.K
            for i in range(self.K):
                # probability of instance x given the gaussian[i], this multiplied by the prior probability mix[i]
                weights[i] = self.conditionalProbability(x, i)
            
            # to avoid nan
            weights = np.nan_to_num(weights)
            
            # add into loglike
            self.loglike += np.log(np.sum(weights))
            
            # sum of probabilities, in other words, the sum of instance x belongs to all clusters, on the final is equal to one
            den = np.sum(weights)
            
            # completing the theorem of bayes for all clusters 
            weights /= den
            
            # returning the weights of all instances for all clusters
            matrixWeights.append(weights)
            
        return np.asarray(matrixWeights)
    
    def Mstep(self, matrixW):
        '''
        method to maximize the probabilities of the gaussians
        '''
        
        # variable to store the density of each gaussian
        self.dens = [None] * self.K
        
        # updating each gaussian
        for i in range(self.K):
            
            # dividing the probabilities to each cluster
            wgts = matrixW[:,i]
            wgts = np.nan_to_num(wgts)
            
            # this variable is going to responsible to storage the sum of probabilities for each cluster
            dens = np.sum(wgts)
            if(dens == 0):
                dens = 0.01
            
            # compute new means
            self.gaussians[i].mu = np.sum(prob*inst/dens for(prob, inst) in zip(wgts, self.train_input))

            # compute new sigma (covariance)
            def covProb(mu, wgts, dens):
                '''
                submethod to update the covariance
                '''
                mu = np.transpose([mu])
                cvFinal = 0
                for i in range(len(wgts)):
                    dt = np.transpose([self.train_input[i]])
                    cv = np.dot(np.subtract(mu, dt), np.transpose(np.subtract(mu, dt)))
                    cv = wgts[i]*cv/dens
                    if(i==0):
                        cvFinal = cv
                    else:
                        cvFinal = np.add(cvFinal, cv)
                return cvFinal
                
            self.gaussians[i].sigma = covProb(self.gaussians[i].mu, wgts, dens)
                
            # compute new mix
            self.dens[i] = dens
            self.mix[i] = dens/len(self.train_input) 
    
    def probs(self, x):
        '''
        method to return the probs of an example for each gaussian
        '''

        z = [0]*self.K
        for i in range(self.K):
            z[i] = self.conditionalProbability(x, i)
        dens = np.sum(z)
            
        z = [0]*self.K
        for i in range(self.K):
            z[i] = (self.conditionalProbability(x, i))/dens
            
        return z
    
    def predictionProb(self, x):
        '''
        method to calculate the probability of a variable x to be on the distribution created
        :param: x: float - variable that we need to know the probability
        :return: the probability of the given variable
        '''
        y = 0
        for i in range(self.K):
            y += self.conditionalProbability(x, i)
        return y
    
    def predict(self, x):
        '''
        method to predict the class for several patterns
        :param: x: pattern
        :return: the respective label for x
        '''

        # to predict multiple examples
        if(len(x.shape) > 1):
            # returning all labels
            return [self.predict_one(pattern) for pattern in x]
        # to predict only one example
        else:
            return self.predict_one(x)
    
    def chooseBestModel(self, train_input, type_selection, Kmax, restarts, iterations):
        '''
        methodo to train several gmms and return the gmm with the best loglike
        :param: train_input: data that will be used to train the model
        :param: type_selection: name of prototype selection metric
        :param: Kmax: number max of gaussians to test
        :param: restarts: integer - number of restarts
        :param: iterations: integer - number of iterations to trains the gmm model
        :return: the best gmm model
        '''
        
        # creating the first gaussian
        gmm = copy.deepcopy(self)
        gmm.fitClustering(train_input, 1)
        gmm.trainEM(iterations)
        
        # evaluating the model
        bestMetric = ps.prototype_metric(type_selection, gmm.loglike, gmm.p, gmm.n)
        bestGmm = gmm
        
        # for to test several values of K
        for k in range(2, Kmax+1):
            
            # for to restart the models
            for _ in range(restarts):
                #print('K[', k, '] restart[', i, ']')
                gmm = copy.deepcopy(self)
                gmm.fitClustering(train_input, k)
                gmm.trainEM(iterations)
                
                # evaluating the model
                metric = ps.prototype_metric(type_selection, gmm.loglike, gmm.p, gmm.n)
                
                # condition to store the best model
                if(metric < bestMetric):
                    bestMetric = metric
                    bestGmm = gmm
            
        # the best model
        return bestGmm
    
    def printstats(self):
        '''
        method to print the parameters of gaussians
        '''
        if(self.ismatrix):
            print('-----------------------------------new it---------------------------------------')
            for i in range(self.K):
                print("cluster [", i, "]: mix =", self.mix[i], "mu =", self.gaussians[i].mu, "sigma =", self.gaussians[i].sigma)
        else:
            print('-----------------------------------new it---------------------------------------')
            for i in range(self.K):
                print("cluster [", i, "]: mix =", self.mix[i], "mu =", self.gaussians[i].mu, "sigma =", self.gaussians[i].sigma)
    
    def plotGmm(self, t, show=True):
        
        # defining the colors of each gaussian
        colors = cm.rainbow(np.linspace(0, 1, self.L))
        marks = ["^", "o", '+', ',']
        
        # creating the image
        plt.subplot(111)
            
        # receiving each observation per class
        classes = []
        for i in range(self.L):
            aux = []
            for j in range(len(self.train_target)):
                if(self.train_target[j] == i):
                    aux.append(self.train_input[j])
            classes.append(np.asarray(aux))
                
        # plotting each class
        classes = np.asarray(classes)
        for i in range(self.L):
            plt.scatter(classes[i][:,0],
                        classes[i][:,1],
                        color = colors[i],
                        #marker = marks[i],
                        label = 'class '+str(i)) 
            
        # plotting the gaussians under the dataset
        for i in range(len(self.gaussians)):
            c = colors[self.gaussians[i].label]

            # plotting the number of gaussian
            plt.text(self.gaussians[i].mu[0], self.gaussians[i].mu[1], "Gaussian ["+str(i)+ "]")
                                
            # plotting the gaussian
            self.draw_ellipse(self.gaussians[i].mu, self.gaussians[i].sigma, c)
            
        # definindo o titulo e mostrando a imagem
        plt.title('GMM - time: ' +str(t))
        plt.legend()
        if(show):
            plt.show()
    
    def plotGmmClustering(self, bestGMM):
        
        # defining the colors of each gaussian
        colors = cm.rainbow(np.linspace(0, 1, len(self.gaussians)))
        
        if(self.ismatrix):
            # creating the image
            imagem = plt.subplot(111)
            
            # creating the colours of each point
            indexColors = [np.argmax(self.matrixWeights[i]) for i in range(len(self.matrixWeights))]
            # plotting the dataset
            imagem.scatter(bestGMM.train_input[:,0], bestGMM.train_input[:,1], c=colors[indexColors])
            
            # plotting the gaussians under the dataset
            for i in range(bestGMM.K):
                self.draw_ellipse(bestGMM.gaussians[i].mu, bestGMM.gaussians[i].sigma, colors[i])
            
            # definindo o titulo e mostrando a imagem
            plt.title('GMM')
            plt.show()
            
        else:
            #mixture
            x = np.linspace(min(self.train_input), max(self.train_input), len(self.train_input))
            sns.distplot(self.train_input, bins = round(len(self.train_input)/4), kde=False, norm_hist=True)
            g_both = [bestGMM.pdf(e) for e in x]
            plt.plot(x, g_both, label='gaussian mixture');
            plt.legend();
            plt.show()

    def plotGmmTrainTest(self, bestGMM, accur_train, accur_test, show=True):
        
        # defining the colors of each gaussian
        colors = cm.rainbow(np.linspace(0, 1, bestGMM.L))
        marks = ["^", "o", '+', ',']
        
        # creating the image
        plt.subplot(111)
            
        # receiving each observation per class
        classes = []
        for i in range(bestGMM.L):
            aux = []
            for j in range(len(bestGMM.train_target)):
                if(bestGMM.train_target[j] == i):
                    aux.append(bestGMM.train_input[j])
            classes.append(np.asarray(aux))
                
        # plotting each class
        classes = np.asarray(classes)
        for i in range(bestGMM.L):
            plt.scatter(classes[i][:,0],
                        classes[i][:,1],
                        color = colors[i],
                        marker = marks[i],
                        label = 'class '+str(i)) 
            
        # plotting the gaussians under the dataset
        for i in range(bestGMM.K):
            c = colors[bestGMM.gaussians[i].label]
            # plotting the gaussian
            self.draw_ellipse(bestGMM.gaussians[i].mu, bestGMM.gaussians[i].sigma, c)
            
        texto = ("Train accuracy: %.2f - Test accuracy: %.2f " % (accur_train, accur_test))
            
        plt.annotate(texto,
                xy=(0.5, 0.15), xytext=(0, 0),
                    xycoords=('axes fraction', 'figure fraction'),
                    textcoords='offset points',
                    size=10, ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))
            
        # definindo o titulo e mostrando a imagem
        plt.title(self.NAME)
        plt.legend()
        if(show):
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
            
    def animation(self, it):
        '''
        method to call an animation
        :param: it: quantity of iterations necessary to simule 
        '''
        # creating the figure that will plot the gmm
        fig = plt.figure()
        img0 = fig.add_subplot(3, 2, (2, 6))
        img1 = fig.add_subplot(3, 2, 1)
        img2 = fig.add_subplot(3, 2, 3)
        img3 = fig.add_subplot(3, 2, 5)
    
        # defining the colors of each gaussian
        colors = cm.rainbow(np.linspace(0, 1, len(self.gaussians)))
        
        # variable to store the evolution of loglikelihood
        self.listLoglike = []
        self.listBic = []
        self.listAic = []
        
        def update(i):
            '''
            method to call one plot
            '''
             
            print("[", i, "]")
            #self.printstats()

            # erasing the img to plot a new figure
            img0.clear()
            img1.clear()
            img2.clear()
            img3.clear()
            
            #plotting the metrics 
            img1.plot(self.listLoglike, label='loglikelihood', color = 'r')
            img1.legend()
            img2.plot(self.listBic, label='BIC', color = 'g')
            img2.legend()
            img3.plot(self.listAic, label='AIC')
            img3.legend()
            
            #ploting the points
            img0.set_title(str('GMM - it: %d' % i))
            # creating the colours of each point
            indexColors = [np.argmax(self.matrixWeights[i]) for i in range(len(self.matrixWeights))]
            # plotting the dataset
            img0.scatter(self.train_input[:,0], self.train_input[:,1], c=colors[indexColors], label = 'dataset')
            # plotting the gaussians under the dataset
            for i in range(len(self.gaussians)):
                self.draw_ellipse(self.gaussians[i].mu, self.gaussians[i].sigma, colors[i], img0)
                    
            
            # training the mixture model
            self.Mstep(self.Estep())
            
        # function that update the animation
        _ = anim.FuncAnimation(fig, update, frames=it, repeat=False)
        plt.show()
        
def main():
    print('')
   
if __name__ == "__main__":
    main()        
            
            
            
            
        
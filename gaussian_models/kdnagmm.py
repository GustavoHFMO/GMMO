'''
Created on 19 de out de 2018
@author: gusta
'''

from sklearn.model_selection import train_test_split
from data_streams.static_datasets import Datasets
from gaussian_models.gmm_super import GMM_SUPER
from sklearn.neighbors import NearestNeighbors
from gaussian_models.gmm import GaussianMixture
from gaussian_models.gaussian import Gaussian
import numpy as np

class KDNAGMM(GMM_SUPER):
    def __init__(self, ruido, remocao, adicao, erro, kmax=None):
        self.ruido = ruido 
        self.remocao = remocao
        self.adicao = adicao
        self.erro = erro
        self.Kmax = kmax
        self.NAME = 'KDNAGMM'
    
    def kDN(self, x, y, n_vizinhos):
        '''
        Metodo para computar o grau de dificuldade de cada observacao em um conjunto de dados
        :param: x: padroes dos dados
        :param: y: respectivos rotulos
        :param: n_vizinhos: quantidade de vizinhos na regiao de competencia
        :return: dificuldades: vetor com a probabilidade de cada instancia 
        '''
        
        # instanciando os vizinhos mais proximos
        nbrs = NearestNeighbors(n_neighbors=n_vizinhos+1, algorithm='ball_tree').fit(x)
        
        # variavel para salvar as probabilidades
        dificuldades = []
        
        # for para cada instancia do dataset
        for i in range(len(x)):
            
            # computando os vizinhos mais proximos para cada instancia
            _, indices = nbrs.kneighbors([x[i]])
            
            # verificando o rotulo dos vizinhos
            cont = 0
            for j in indices[0]:
                if(j != i and y[j] != y[i]):
                    cont += 1
                    
            # computando a porcentagem
            dificuldades.append(cont/(n_vizinhos+1))
        
        return dificuldades
    
    def easyInstances(self, x, y, k, limiar):
        '''
        Metodo para retornar um subconjunto de validacao apenas com as instacias faceis
        :param: x: padroes dos dados
        :param: y: respectivos rotulos
        :return: x_new, y_new: 
        '''
        
        # computando as dificulades para cada instancia
        dificuldades = self.kDN(x, y, k)
        
        # to guarantee that will be there at least one observation
        cont = 0
        for i in dificuldades:
            if(i > limiar):
                cont += 1
        if(cont <= (len(dificuldades)/3)):
            limiar = 1
        
        # variaveis para salvar as novas instancias
        x_new = []
        y_new = []
         
        # salvando apenas as instancias faceis
        for i in range(len(dificuldades)):
            if(dificuldades[i] < limiar):
                x_new.append(x[i])
                y_new.append(y[i])
                
        return np.asarray(x_new), np.asarray(y_new)
    
    def fit(self, train_input, train_target, type_selection='AIC', Kmax=4, restarts=1, iterations=5, k=5, limiar=0.7):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: type_selection: name of prototype selection metric. Default 'AIC'
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # to experiment
        if(self.Kmax != None):
            Kmax = self.Kmax
        
        # storing the patterns
        if(self.ruido):
            self.train_input, self.train_target = self.easyInstances(train_input, train_target, k, limiar)
        else:
            self.train_input, self.train_target = train_input, train_target

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
    
    def strategy(self, y_true, y_pred, x, W, i):
        '''
        method to update an gaussian based on error 
        '''
        
        condition = False
        if(self.erro and y_true != y_pred):
            condition = True
        else:
            if(i % 5 == 0):
                condition = True
            
        if(condition):
            
            # adjusting the window
            W = np.asarray(W)
                    
            # updating the data for train and target
            self.train_input = W[:,0:-1]
            self.train_target = W[:,-1]
            
            # find the nearest gaussian
            flag, gaussian = self.nearestGaussian(x, y_true)
            
            # condition
            if(flag):
                # update the nearest gaussian
                self.updateGaussianIncremental(x, gaussian)

            else:
                if(self.adicao):
                    # create a new gaussian
                    self.createGaussian(x, y_true)

            if(self.remocao):
                # gmm maintenance
                self.removeGaussians()
        
            # returning the new prediction for later analysis 
            return self.predict(x)
         
        return None

    def nearestGaussian(self, x, y):
        '''
        method to find the nearest gaussian of the observation x
        :x: observation 
        :y: label
        '''
        
        # receiving the gaussian with more probability for the pattern
        z = [0] * len(self.gaussians)
        for i in range(len(self.gaussians)):
            if(self.gaussians[i].label == y):
                z[i] = self.gaussians[i].pdf_vector(x)

        # nearest gaussian
        gaussian = np.argmax(z)
        
        # condition to create or update gaussian
        if(z[gaussian] > 0.0):
            return True, gaussian
        else:
            return False, gaussian
        
    def removeGaussians(self):
        '''
        method to remove obsolete gaussians
        '''
        
        # ammount of gaussians per class
        class_gaussians = [self.gaussians[i].label for i in range(len(self.gaussians))]
        labels, ammount = np.unique(class_gaussians, return_counts=True)
            
        # to search obsolete gaussians
        erase = []
        for i in range(len(labels)):
            for j in range(len(self.gaussians)):
                if(ammount[i] > 1 and self.gaussians[j].label == labels[i] and self.mix[j] == 0.0):
                    erase.append(j)
        
        # to remove obsolete gaussians
        for i in sorted(erase, reverse=True):
            del self.gaussians[i]
            del self.mix[i]
            self.K -= 1
    
    def createGaussian(self, x, y):
        '''
        method to create a new gaussian
        :x: observation 
        :y: label
        '''
        
        # mu
        mu = x
        
        # covariance
        cov = (0.5**2) * np.identity(len(x))
        
        # label
        label = y
        
        # new gaussian
        g = Gaussian(mu, cov, label)
        
        # adding the new gaussian in the system
        self.gaussians.append(g)
        
        # adding the new mix
        self.mix.append(1)
        
        # adding the dens
        self.dens.append(1)
        
        # adding 
        self.K += 1
        
        # updating the density of all gaussians
        self.updateLikelihood(x)
        
        # updating the weights of all gaussians
        self.updateWeight()
        
    def updateGaussianIncremental(self, x, gaussian):
        '''
        method to update the nearest gaussian of x
        :x: the observation that will be used to update a gaussian
        :gaussian: the number of gaussian that will be updated  
        '''

        # updating the likelihood of all gaussians for x
        self.updateLikelihood(x)
                
        # storing the old mean
        old_mean = self.gaussians[gaussian].mu
        
        teste = self.updateMean(x, gaussian)
        if(np.any(np.isnan(teste))):
            print()
        
        # updating the mean
        self.gaussians[gaussian].mu = self.updateMean(x, gaussian)

        # updating the covariance        
        self.gaussians[gaussian].sigma = self.updateCovariance(x, gaussian, old_mean)
        
        # updating the gaussian weights
        self.updateWeight()
        
    def updateLikelihood(self, x):
        '''
        method to update the parameter cver
        :param: x: new observation
        '''
        
        # updating the loglikelihood
        for i in range(len(self.gaussians)):
            self.dens[i] = self.dens[i] + self.posteriorProbability(x, i)
        
    def updateWeight(self):
        '''
        Method to update the mix
        '''
        
        sum_dens = np.sum(self.dens)
        if(sum_dens == 0.0): sum_dens = 0.01
        
        for i in range(len(self.gaussians)):
            self.mix[i] = self.dens[i]/sum_dens
    
    def updateMean(self, x, i):
        '''
        Method to update the mean of a gaussian i
        return new mean
        '''
        
        part1 = self.posteriorProbability(x, i)/self.dens[i]
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
        part5 = self.posteriorProbability(x, i)/self.dens[i]
        part6 = np.subtract(x, self.gaussians[i].mu)
        part7 = np.transpose(part6)
        part8 = np.dot(part6, part7)
        part9 = np.subtract(part8, part0)
        part10 = np.dot(part5, part9)
        
        #final
        covariance = np.add(part4, part10) 
        
        return covariance
    
def main():
   
    dt = Datasets()
    X, y = dt.chooseDataset(5)
    
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
    gmm = KDNAGMM(ruido=True, remocao=True, adicao=True, erro=True, kmax=10)
    
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
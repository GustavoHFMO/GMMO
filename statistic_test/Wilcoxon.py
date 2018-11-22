#-*- coding: utf-8 -*-
'''
Created on 10 de nov de 2018
@author: gusta
'''

import numpy as np
from statistic_test.LerDados import Ler_dados

from scipy.stats import wilcoxon

class WilcoxonTest():
    def __init__(self, labels, data):
        self.labels = labels
        self.data = data

    def do(self, reverse=False):
        #computando o wilcoxon
        _, p_value = wilcoxon(self.data[0] - self.data[1])

        # computando a media
        mean0 = np.mean(self.data[0])
        mean1 = np.mean(self.data[1])
        
        # labels para print        
        label0 = self.labels[0]
        label1 = self.labels[1]
                    
        if(p_value < 0.05):
            if(mean0 > mean1):
                label0 = self.labels[1]
                label1 = self.labels[0]
                mean0 = np.mean(self.data[1])
                mean1 = np.mean(self.data[0])
                
            print("the algorithm: "+label0+" ("+str(mean0)+") is the less than: "+label1+" ("+str(mean1)+")")
        else:
            print("the algorithms are statistically equal!")
            print("the algorithm: "+label0+": "+str(mean0)+" \n"+label1+": "+str(mean1))
        print("paired wilcoxon-test p-value: ", p_value)
        
    def Exemplo_executavel(self):
        #acuracias dos modelos, cada coluna é um modelo
        data1 = np.asarray([3.88, 5.64, 5.76, 4.25, 5.91, 4.33])
        data2 = np.asarray([30.58, 30.14, 16.92, 23.19, 26.74, 10.91])
                            
        #label dos modelos, cada coluna é um modelo
        names_alg = ["alg1", 
                     "alg2"]
        
        wc = WilcoxonTest(names_alg, [data1, data2])
        wc.do()

def main():
    
    tbt = Ler_dados()
    i = 6
    arquivo = ['circles-accuracy', 
               'sine1-accuracy', 
               'sine2-accuracy',
               'virtual_5changes-accuracy', 
               'virtual_9changes-accuracy',
               'SEA-accuracy', 
               'SEARec-accuracy']
    
    print(arquivo[i])
    caminho_arquivo = 'E:/Workspace2/GMMO/projects/modificacoes/'+arquivo[i]+'.xls'
    labels, acuracias = tbt.obter_dados_arquivo(caminho_arquivo, [1, 11], [0, 7])
    #['com ruido', 'sem remocao', 'sem adicao', 'tudo', 'nada', 'atualizacao constante', 'atualizacao constante sem remocao', 'sem deteccao']
    #print(labels)
    
    names_alg = ['atualizacao constante', 'tudo']
    data1 = acuracias[labels.index(names_alg[0])]
    data2 = acuracias[labels.index(names_alg[1])]
        
    wc = WilcoxonTest(names_alg, [data1, data2])
    wc.do(reverse=True)
    
if __name__ == '__main__':
    main()    
        
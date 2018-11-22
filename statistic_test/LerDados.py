#-*- coding: utf-8 -*-
'''
Created on 10 de nov de 2018
@author: gusta
'''

import numpy as np

class Ler_dados():
    pass

    def obter_dados_arquivo(self, caminho_tabela, linhas, colunas):
        '''
        metodo para gerar plots do teste estatistico
        :param: caminho_tabela: string, referente ao caminho que a tabela se encontra
        :param: linhas: vetor inteiro, contendo o inicio da linha e o final dos dados para serem usados no teste
        :param: colunas: vetor inteiro, contendo o inicio da coluna e o final dos dados para serem usados no teste
        '''
        
        # importando lib que abre o arquivo
        import xlrd
        # abrindo um workbook e copiando ele em uma variavel auxiliar
        book = xlrd.open_workbook(caminho_tabela)
        # obtendo a quantidade de folhas 
        i = len(book.sheet_names())
        # abrindo a folha existente
        #sh = book.sheet_by_index(i-1)
        sh = book.sheet_by_index(0)
            
        # obtendo a quantidade linhas dentro da folha
        linha_inicial = linhas[0]
        linha_final = linhas[1]
        
        # obtendo a quantidade colunas dentro da folha
        coluna_inicial = colunas[0]
        coluna_final = colunas[1]
            
        # variavel para salvar a primeira coluna das caminho_tabela
        labels = []
        acuracias = []
        
        # for para a quantidade de colunas
        for j in range(coluna_inicial, coluna_final+1):
            # for para percorrer as linhas de cada coluna
            for k in range(0, 1):
                # copiando o valor referente a linha e coluna passada
                valor = sh.cell_value(rowx=k, colx=j)
                labels.append(valor)
                
                
        # for para a quantidade de colunas
        for j in range(coluna_inicial, coluna_final+1):
            # variavel para salvar os valores de cada coluna
            valores = []
            # for para percorrer as linhas de cada coluna
            for k in range(linha_inicial, linha_final):
                # copiando o valor referente a linha e coluna passada
                valor = sh.cell_value(rowx=k, colx=j)
                valores.append(valor)
            # salvando o conjunto de acuracias    
            acuracias.append(np.asarray(valores))
        # convertendo a lista final em um array
        acuracias = np.asarray(acuracias)
            
        return labels, acuracias

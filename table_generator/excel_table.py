#-*- coding: utf-8 -*-
from xlwt import Workbook
from xlwt.ExcelFormula import Formula
import xlrd
import xlwt
import numpy as np

class Tabela_excel():
    def __init__(self):
        '''
        classe para escrever dados em um arquivo xlsx
        '''
        
        self.wb = Workbook(style_compression=2)
        self.sheets = []
        self.nome_tabela = []
        self.ncols = 0
    
    def Estilo_cabecalho(self):
        '''
        metodo para colocar um estilo no cabecalho
        '''
        
        #Fonte do cabecalho
        font_cabecalho = xlwt.Font()
        font_cabecalho.name = "Times New Roman"
        font_cabecalho.bold = True
        self.estilo_cabecalho = xlwt.XFStyle()
        self.estilo_cabecalho.font = font_cabecalho
        #self.estilo_cabecalho.num_format_str = '0.00%'
        
        return self.estilo_cabecalho
    
    def Estilo_texto(self):
        '''
        metodo para colocar um estilo no texto comum
        '''
        
        #Fonte do texto
        font_texto = xlwt.Font()
        font_texto.name = "Times New Roman"
        font_texto.bold = False
        self.estilo_texto = xlwt.XFStyle()
        self.estilo_texto.font = font_texto
        
        return self.estilo_texto

    def Gerar_nome(self, nome):
        '''
        metodo para gerar o nome do arquivo xlsx
        :param nome: string com o nome do futuro arquivo xlsx
        '''
        
        #data = datetime.now()
        #self.nome_tabela = str(nome)+ ' ' +str(data.strftime("%A %d %B %Y %H %M %S")) + '.xls'
        self.nome_tabela = str(nome) + '.xls'
                
    def Criar_tabela(self, nome_tabela, folhas, cabecalho = None, largura_col = None):
        '''
        metodo para criar o arquivo xlsx com a quantidade de folhas especificas
        :param nome: string com o nome do futuro arquivo xlsx
        :param folhas: lista com o nome e a quantidade de folhas que o arquivo vai possuir
        :param cabecalho: cabecalho para ser colocado no inicio de cada folha
        :param largura_col: largura de cada coluna escrita
        '''
        
        if(cabecalho != None):
            self.ncols = len(cabecalho)
        
        self.Gerar_nome(nome_tabela)
        
        #criando as folhas
        for e in folhas:
            self.sheets.append(self.wb.add_sheet(e))

        #obtendo os estilos da tabela
        estilo_cabecalho = self.Estilo_cabecalho()
        
        
        #criando os cabecalhos para as folhas
        if(cabecalho != None):
            for folha in self.sheets:
                for x, i in enumerate(cabecalho):
                    folha.write(0,x, i, estilo_cabecalho)
                    folha.col(x).width = largura_col
        
        elif(largura_col != None):
            for folha in self.sheets:
                for i in range(10):
                    folha.col(i).width = largura_col
            
            
        self.wb.save(self.nome_tabela)

    def ler(self, arq_xls, folha, linha, coluna):
        '''
        método para ler um valor de uma celula especifica
        :param: arq_xls: nome do arquivo
        :param: folha: folha em que o dado se encontra
        :param: linha: linha referente
        :param: coluna: coluna referente
        :return: celula buscada
        '''
        
        book = xlrd.open_workbook(arq_xls)
        #print("número de abas: ", book.nsheets)
        #print("Nomes das Planilhas:", book.sheet_names())
        sh = book.sheet_by_index(folha)
        #print(sh.name, sh.nrows, sh.ncols)
        #print("Valor da celula D30 é ", sh.cell_value(rowx=10, colx=4))
        return sh.cell_value(rowx=linha, colx=coluna)
    
    def Adicionar_coluna(self, num_sheet, coluna, valores):
        '''
        metodo para escrever um grupo de valores em uma coluna
        :param num_sheet: numero da folha que sera escrita
        :param coluna: coluna na qual o valor deve ser escrito
        :param valor: valor que sera escrevido
        '''
        
        for i in range(len(valores)):
            self.Adicionar_dado(num_sheet, i, coluna, valores[i])
            
    def Adicionar_Sheet_Linha(self, num_sheet, execucao, valores):
        '''
        metodo para escrever os dados em uma linha
        :param num_sheet: numero da folha que sera escrita
        :param execucao: linha na qual o valor deve ser escrito
        :param valores: lista com os valores que serao escritos por coluna
        '''
        
        
        estilo_texto = self.Estilo_texto()
        
        for x, valor in enumerate(valores):
            self.sheets[num_sheet].write(execucao+1, x, valor, estilo_texto)
        
        self.wb.save(self.nome_tabela)
        #print("Salvou!")
        
    def Adicionar_dado(self, num_sheet, linha, coluna, valor):
        '''
        metodo para escrever um dado especifico em uma determinada posicao
        :param num_sheet: numero da folha que sera escrita
        :param coluna: coluna na qual o valor deve ser escrito
        :param linha: linha na qual o valor deve ser escrito
        :param valor: valor que sera escrevido
        '''
        
        #estilo_texto = self.Estilo_texto()
        
        self.sheets[num_sheet].write(linha, coluna, valor)
        
        self.wb.save(self.nome_tabela)
        #print("Salvou!")
    
    def Calcular_Medias_formula(self, qtd_execucoes):
        '''
        metodo para computar a media das colunas no final do arquivo
        :param qtd_execucoes: linha em que as medias serao escrevidas
        '''
        
        for e in self.sheets:
            e.write(qtd_execucoes+1, 0, Formula('AVERAGE(A2:A'+str(qtd_execucoes+1)+')'), self.estilo_cabecalho)
            e.write(qtd_execucoes+1, 1, Formula('AVERAGE(B2:B'+str(qtd_execucoes+1)+')'), self.estilo_cabecalho)
            e.write(qtd_execucoes+1, 2, Formula('AVERAGE(C2:C'+str(qtd_execucoes+1)+')'), self.estilo_cabecalho)
            e.write(qtd_execucoes+1, 3, Formula('AVERAGE(D2:D'+str(qtd_execucoes+1)+')'), self.estilo_cabecalho)
            e.write(qtd_execucoes+1, 4, Formula('AVERAGE(E2:E'+str(qtd_execucoes+1)+')'), self.estilo_cabecalho)
            self.wb.save(self.nome_tabela)
            
        self.wb.save(self.nome_tabela)
        print("Salvou a tabela!")
        
    def Calcular_Medias(self, qtd_execucoes):
        '''
        metodo para computar a media das colunas no final do arquivo
        :param qtd_execucoes: linha em que as medias serao escrevidas
        '''
        
        for e in range(len(self.sheets)):
            for j in range(self.ncols):
                media = []
                for i in range(qtd_execucoes):
                    valor = self.ler(self.nome_tabela, e, i+1, j)
                    media.append(valor)
                self.sheets[e].write(qtd_execucoes+1, j, np.mean(media), self.estilo_cabecalho)
                self.wb.save(self.nome_tabela)
        
        print("Salvou a tabela!")
    
    def Calcular_Medias3(self, qtd_execucoes):
        '''
        metodo para computar a media das colunas no final do arquivo
        :param qtd_execucoes: linha em que as medias serao escrevidas
        '''
        book = xlrd.open_workbook(self.nome_tabela)
        
        for e in range(len(self.sheets)):
            sh = book.sheet_by_index(e)
            for j in range(self.ncols-1):
                media = []
                for i in range(qtd_execucoes):
                    valor = sh.cell_value(rowx=i+1, colx=j+1)
                    media.append(valor)
                self.sheets[e].write(qtd_execucoes+1, j+1, np.mean(media), self.estilo_cabecalho)
                self.wb.save(self.nome_tabela)
        
        print("Salvou a tabela!")

def main():
    tabela = Tabela_excel()
    nome = "teste"
    folhas = ["sheet1", "sheet2", "sheet3", "sheet4"]
    tabela.Criar_tabela(nome, folhas, cabecalho=None, largura_col=5000)
    
    tabela1 = Tabela_excel()
    nome = "teste1"
    folhas = ["sheet1", "sheet2", "sheet3", "sheet4"]
    tabela1.Criar_tabela(nome, folhas, cabecalho=None, largura_col=5000)
    
    tabela2 = Tabela_excel()
    nome = "teste2"
    folhas = ["sheet1", "sheet2", "sheet3", "sheet4"]
    tabela2.Criar_tabela(nome, folhas, cabecalho=None, largura_col=5000)
    
    for j in range(4):
        for i in range(100):
            print('[',j,'][',i,']')
            tabela.Adicionar_coluna(j, i, [np.random.random() for _ in range(100)])
            tabela1.Adicionar_coluna(j, i, [np.random.random() for _ in range(100)])
            tabela2.Adicionar_coluna(j, i, [np.random.random() for _ in range(100)])
    
    '''
    qtd_execucoes = 10
    
    for execucao in range(qtd_execucoes):
        print(execucao)
        
        for folha in range(len(folhas)):
            tabela.Adicionar_Sheet_Linha(folha, execucao, lista)
            
    tabela.Calcular_Medias(qtd_execucoes)
    '''
    
    
    
    
if __name__ == "__main__":
    main()



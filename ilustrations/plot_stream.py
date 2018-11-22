'''
Created on 20 de ago de 2018
@author: gusta
'''

from streams.readers.arff_reader import ARFFReader
from data_streams.adjust_labels import Adjust_labels
ad = Adjust_labels()
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plotData(i, data, labels, fig=None):
    '''
    method to plot the current data inside a window
    :param: labels: the respective labels of data
    :param: data: the data that will be plotted
    '''
    
    # sorting colors
    clrs = cm.rainbow(np.linspace(0, 1, len(labels)))
    
    # transforming a list into a array
    data = np.asarray(data)

    # obtaining the data labels
    index = ad.targetStream(labels, data)
    
    if(fig==None):
        # plotting the dataset
        plt.title(i)
        plt.scatter(data[:,0], data[:,1], c=clrs[index])
        plt.show()
    
    else:
        # plotting the dataset
        plt.title(i)
        fig.scatter(data[:,0], data[:,1], c=clrs[index])
            
def slidingWindow(W, x):
    '''
    method to slide the window under the example
    :param: W: window that will be updated 
    :param: x: example that will be inserted 
    '''
    
    aux = [None] * len(W)
    aux[0:-1] = W[1:]
    aux[-1] = x
    
    return aux
 

# Loading an arff file

def run():
    
    # filling the initial window
    m = 400
    W = stream_records[0:m]
    
    # initial plot
    plotData(W)
    
    # stream
    for i, x in enumerate(stream_records[m:]):
    
        print(i)
            
        # sliding window under the stream
        W = slidingWindow(W, x)
    
        #plotting the window
        if(i % 5 == 0):
            plotData(W)

def animation():
        '''
        method to call an animation
        :param: it: quantity of iterations necessary to simule 
        '''
    
        # creating the figure
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
           
        def update(i):
            
            # cleaning the figure
            ax.clear()
            
            # printing iteration
            print(i)
                
            i+= 1400
            tam = 300
            # sliding window under the stream
            W = stream_records[i:tam+i]
        
            # plotting W
            plotData(i+tam, W, labels, ax)
           
            
        # function that update the animation
        _ = anim.FuncAnimation(fig, update, repeat=True)
        plt.show()

def staticPlot(name, stream_records, labels, m):
    '''
    method to plot several time steps of stream
    '''
    
    # transforming in array
    stream_records = np.asarray(stream_records)

    # creating the location to each plot
    fig = plt.figure()
    fig.suptitle(name, fontsize=11, fontweight='bold')
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    ax4 = fig.add_subplot(3, 3, 4)
    ax5 = fig.add_subplot(3, 3, 5)
    ax6 = fig.add_subplot(3, 3, 6)
    ax7 = fig.add_subplot(3, 3, 7)
    ax8 = fig.add_subplot(3, 3, 8)
    
    
    if(len(stream_records) > 8000):
        ax9 = fig.add_subplot(3, 3, 9)

    # sorting colors
    clrs = cm.rainbow(np.linspace(0, 1, len(labels)))
    
    #plot 1
    point = 0
    data1 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data1)
    ax1.scatter(data1[:,0], data1[:,1], c=clrs[index])
    ax1.set_title("time: "+str(point))
    
    #plot 2
    point = 1000
    data2 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data2)
    ax2.scatter(data2[:,0], data2[:,1], c=clrs[index])
    ax2.set_title("time: "+str(point))
    
    #plot 3
    point = 2000
    data3 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data3)
    ax3.scatter(data3[:,0], data3[:,1], c=clrs[index])
    ax3.set_title("time: "+str(point))
    
    #plot 4
    point = 3000
    data4 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data4)
    ax4.scatter(data4[:,0], data4[:,1], c=clrs[index])
    ax4.set_title("time: "+str(point))
    
    #plot 5
    point = 4000
    data5 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data5)
    ax5.scatter(data5[:,0], data5[:,1], c=clrs[index])
    ax5.set_title("time: "+str(point))
    
    #plot 6
    point = 5000
    data6 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data6)
    ax6.scatter(data6[:,0], data6[:,1], c=clrs[index])
    ax6.set_title("time: "+str(point))
    
    #plot 7
    point = 6000
    data7 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data7)
    ax7.scatter(data7[:,0], data7[:,1], c=clrs[index])
    ax7.set_title("time: "+str(point))
    
    #plot 8
    point = 7000
    data8 = stream_records[point:point+m]    
    index = ad.targetStream(labels, data8)
    ax8.scatter(data8[:,0], data8[:,1], c=clrs[index])
    ax8.set_title("time: "+str(point))
    
    if(len(stream_records) > 8000):
        #plot 9
        point = 8000
        data9 = stream_records[point:point+m]    
        index = ad.targetStream(labels, data9)
        ax9.scatter(data9[:,0], data9[:,1], c=clrs[index])
        ax9.set_title("time: "+str(point))
    
    # show
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#labels, attributes, stream_records = ARFFReader.read("../data_streams/_synthetic/virtual.arff")

vr = 0
dataset = "circles"
labels, attributes, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset+"/"+dataset+"_"+str(vr)+".arff")
#animation()
staticPlot("Dataset: "+dataset, stream_records, labels, 300)

'''
labels, attributes, stream_records = ARFFReader.read("../data_streams/_synthetic/circles.arff")
staticPlot('Dataset: circles', stream_records, labels, 300)

labels, attributes, stream_records = ARFFReader.read("../data_streams/_synthetic/sine1.arff")
staticPlot('Dataset: sine1', stream_records, labels, 300)

labels, attributes, stream_records = ARFFReader.read("../data_streams/_synthetic/sine2.arff")
staticPlot('Dataset: sine2', stream_records, labels, 300)
'''

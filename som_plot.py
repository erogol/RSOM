import matplotlib.pyplot as plt
from pylab import plot,axis,show,pcolor,colorbar,bone

# data need to be 2 dim
def som_plot_scatter(W,X,A):
	#plt.plot(blobs[0],'bo')
    cof = 5000 / A.max();
    A = (A * cof)
    plt.figure(7)
    plt.scatter(X[:,0], X[:,1], color='red')
    plt.scatter(W[:,0],W[:,1],color='blue',s=A.T/10 ,edgecolor='none')
    for count, i in enumerate(W):
        plt.annotate(count, xy = i, xytext = (0, 0), textcoords = 'offset points')
    plt.show()

def som_plot_outlier_scatter(W,X,unit_saliency, inst_saliency, A):
    plt.scatter(X[inst_saliency == True, 0], X[inst_saliency == True,1], color='yellow')
    plt.scatter(X[inst_saliency == False, 0], X[inst_saliency == False, 1], color = 'orange')
    plt.scatter(W[unit_saliency == True ,0],W[unit_saliency == True ,1],color='blue',s=20 ,edgecolor='none')
    plt.scatter(W[unit_saliency == False ,0],W[unit_saliency == False ,1],color='red',s=20 ,edgecolor='none')
    #for count, i in enumerate(W):
    #    plt.annotate(count, xy = i, xytext = (0, 0), textcoords = 'offset points')
    plt.show()
    
def som_plot_mapping(distance_map):
    bone()
    pcolor(distance_map.T) # plotting the distance map as background
    colorbar()
    #axis([0,som.weights.shape[0],0,som.weights.shape[1]])
    show() # show the figure


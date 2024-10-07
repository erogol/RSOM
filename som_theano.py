from __future__ import division
'''
    Rectifying Self Organazing Maps a.k.a RSOM
    
    RSOM is a clustering and outlier detection method that is predicated with
    old Self Organazing Maps.
    
    It includes Batch and Stochastic learning rules. There are two different
    implementations. One is based on Numpy and tthe other is Theano. If you have
    tall and wide data matrix, we suggest to use Theano version. Otherwise 
    Numpy version is faster. You can also use GPU with Theano but you need to 
    set Theano configurations.
    
    For more detail about RSOM refer to http://arxiv.org/abs/1312.4384
    
    AUTHOR:
        Eren Golge
        erengolge@gmail.com
        www.erengolge.com
'''

"""
TO DO:
-> Try dot product distance instead of Euclidean 
-> Normzalize only updated weight vectors in that epoch
-> compare code with https://github.com/JustGlowing/minisom/blob/master/minisom.py
-> print resulting objective values
-> write bookeeping for best objective value
-> learning rate is already decreasing so radius might be good to keep it constant
-> UPDATE only winners 
"""

import warnings
from random import *
from math import *
import sys
import scipy
import numpy as np
from numpy import linalg
from som_plot import *
import theano
import theano.tensor as T
from theano import function, config, shared, sandbox
from theano import ProfileMode
from collections  import Counter
#from theano import ProfileMode

EPS =  2.2204e-16;

class SOM(object):

    def __init__(self, DATA=None,  num_units = 10, height=None, width=None, \
     alpha_max=0.05, alpha_min=0.001, set_count_activations = True, \
     set_outlier_unit_det = True, set_inunit_outlier_det = True, outlier_unit_thresh = 0.5,\
     inunit_outlier_thresh = 95):
         
        '''
             CONSTRUCTOR PARAMETERS:

                DATA                    --- data matrix with shape nxm n is number of instances and
                                         m is number of variables
                num_units               --- number of som units. This can be changes a bit after
                                         2D lattice shape is computed by eigen heuristic, if its shape
                                         paramters are not given already.
                height                  --- height of the 2D lattice of SOM
                width                   --- width of the 2D lattice of SOM. height * width = num_inst
                alpha_max               --- is the maximum learning rate that is gradually 
                                         decreasing up to alpha_min
                alpha_min               --- is the minimum learning rate attined at the last epoch
                set_count_activations   --- whether count the activation of each unit
                set_outlier_unit_det    --- whether outlier units are detected. If a unit 
                                         is detected as outlier, all of the assigned items signed as outlier as well
                set_inunit_outlier_det  --- wheter in-unit outlier instances are detected
                outlier_unit_thresh     --- default value 0.5 works good for many cases
                inunit_outlier_thresh   --- is the upper whisker percentage.
        '''

        self.X = DATA
        self.num_units = num_units
        if height == None or width == None:
            self._estimate_map_shape()
            self.num_units = self.height * self.width
        else:
            self.height = height
            self.width = width
        
        if self.height * self.width != self.num_units:
            print "Number of units is not conforming to lattice size so it is set num_units = width + heigth"
            self.num_units = self.height * self.width
            print "New number of units : ",self.num_units
            raw_input("Press Enter to continue...")
            
        self.data_dim = DATA.shape[1]

        # normalize data and save mean and std values
        self.data_mean = 0
        self.data_std  = 0
        #self._norm_data()

        # optimization parameters
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        
        self.W = np.random.random((self.num_units , self.data_dim))
        self.W = np.array([v/linalg.norm(v) for v in self.W]) # normalizat   
        
        # book keeping
        self.best_W = self.W
        self.best_W_obj = 0
        
        # unit statistics
        self.set_count_activations = set_count_activations
        self.activations = np.zeros((self.num_units))
        self.set_outlier_unit_det = set_outlier_unit_det
        self.set_inunit_outlier_det = set_inunit_outlier_det  
        self.unit_saliency_coeffs = np.zeros((self.num_units))
        self.unit_saliency = np.ones((self.num_units), dtype=bool)
        self.inst_saliency = np.array(())
        self.outlier_unit_thresh = outlier_unit_thresh
        self.inunit_outlier_thresh = inunit_outlier_thresh
        self.ins_unit_assign = np.array(())
        self.ins_unit_dist = np.array(())
        self.unit_coher = np.array(())

    unit_x = lambda self, index, width : index % width
    unit_y = lambda self, index, width : np.floor( index / width )
    
    def unit_cords(self, index):
        return self.unit_x(index, self.width), self.unit_y(index, self.width)
    
    # Euclidean distance with pre-computed data square X2
    def _euq_dist(self, X2, X):
        return -2*np.dot(self.W, X.T) + (self.W**2).sum(1)[:, None] + X2.T
        
    # Print function for Numpy based optimization functions  
    def _print_cost(self,X2, epoch, num_epoch):
        D = self._euq_dist(X2, self.X)
        print "epoch", epoch, "of", num_epoch, " cost: ", np.linalg.norm(D.min(0), ord=1) / self.X.shape[0]
    
    
    def set_params(self, num_epoch):
        
        '''
            Before starting to learning, all imperative parameters are set regarding
            corresponding epoch. It wastes some additional memory but proposes faster 
            learning speed.
            
            Outputs:
                U --- is a dictionary including all necessary parameter structures
                    
                    U['alphas'] -- learning rates for each epoch
                    U['H_maps'] -- matrix array of neighboorhood masks
                    U['radiuses'] -- neighboor radiuses for each epoch
                    
        '''
        
        U = {'alphas':[], 'H_maps':[], 'radiuses':[]}              
        alphas = [None]*num_epoch       
        H_maps = [None]*num_epoch
        radiuses = [None]*num_epoch

        dist_map = np.zeros((self.num_units, self.num_units))
        radius = np.ceil(1 + floor(min(self.width, self.height)-1)/2)-1
        for u in range(int(self.num_units)):
            #for r in range(1,int(radius)+1,1):  
            dist_map[u,:] = self.find_neighbors(u,self.num_units)
        
        for epoch in range(0,num_epoch,1):
            alpha = self.alpha_max - self.alpha_min
            alpha = alpha * (num_epoch - epoch)
            alpha = alpha / num_epoch + self.alpha_min
            radius = np.ceil(1 + floor(min(self.width, self.height)-1)/2)-1
            radius = radius * (num_epoch - epoch)
            radius = ceil(radius / (num_epoch - 1))-1
            if radius < 0 :
                radius = 0 
            neigh_updt_map = alpha * (1 - dist_map/float((1 + radius))) 
           # neigh_updt_map[dist_map == 0] = 1
            neigh_updt_map[dist_map > radius] = 0 # Optimize this part
            H_maps[epoch] = neigh_updt_map
            alphas[epoch] = alpha
            radiuses[epoch] = radius

        U['alphas'] = alphas
        U['H_maps'] = H_maps
        U['radiuses'] = radiuses
        return U

    def train_stoch(self, num_epoch, verbose =True):
        
        '''
            Numpy based stochastic training where each instance is take individually
            and weight are updatesd in terms of winner neuron. 
            
            Generally faster than Theano version
        '''

        if num_epoch == None:
            num_epoch = 500 * self.num_units # Kohonen's suggestion
            
        U = self.set_params(num_epoch)
        X2 = (self.X**2).sum(1)[:, None]
        
        for epoch in range(num_epoch):
            shuffle_indices = np.random.permutation(self.X.shape[0])
            
            update_rate = U['H_maps'][epoch]
            learn_rate = U['alphas'][epoch]
            win_counts = np.zeros((self.num_units))
            for i in shuffle_indices:
                instance = self.X[i,:]
                D = self._euq_dist(X2[i][None,:], instance[None,:])
                BMU_indx = np.argmin(D)
                
                win_counts[BMU_indx] += 1
                if self.set_count_activations:
                    self.activations[BMU_indx] += 1
                
                self.W  = self.W + learn_rate * update_rate[...,BMU_indx,None]* (instance - self.W)
                ## Normalization is not imperative unless given input instances are normalized
                # self.W = self.W / np.linalg.norm(self.W)

            if verbose and (epoch % 1) == 0:
                self._print_cost(X2, epoch, num_epoch)
            
            if self.set_outlier_unit_det:
                self._update_unit_saliency(win_counts, update_rate, learn_rate)      

        # Normalize activation counts
        if self.set_count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act
        
        self.assing_to_units() # final unit assignments

        if self.set_outlier_unit_det:
            self._find_outlier_units()
        
        if self.set_inunit_outlier_det:
            self._find_inunit_outliers()
            
            
                
    def train_stoch_theano(self, num_epoch = None, verbose =True):
        
        '''
            Theano based stochastic learning
        '''
        
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.filterwarnings("ignore")
        
        if num_epoch == None:
            num_epoch = 500 * self.X.shape[0]
        
        # Symmbol variables
        X = T.dmatrix('X')
        WIN = T.dmatrix('WIN')
        H = T.dmatrix('H')
        
        # Init weights random
        W = theano.shared(self.W, name="W")
        #W = theano.shared(rng.randn(cluster_num, data.shape[1]).astype(theano.config.floatX), name="W")

        # Find winner unit
        D = (W**2).sum(axis=1, keepdims=True) + (X**2).sum(axis=1, keepdims=True).T - 2*T.dot(W, X.T) 
        bmu = (D).argmin(axis=0)
        dist = T.dot(WIN.T, X) - WIN.sum(0)[:, None] * W
        err = D.min(0).norm(1)/X.shape[0]

        update = function([X,WIN, H],outputs=err,updates=[(W, W + T.addbroadcast(H,1)*dist)])
        find_bmu = function([X], bmu)

        # Update
        U = self.set_params(num_epoch)
        for epoch in range(num_epoch):
            update_rate = U['H_maps'][epoch]
            learn_rate = U['alphas'][epoch]
            win_counts = np.zeros((self.num_units))
            shuff_indx = np.random.permutation(self.X.shape[0])
            for i in shuff_indx:
                ins = self.X[i, :][None,:]
                D = find_bmu(ins)
                S = np.zeros([ins.shape[0],self.num_units])
                #S = np.zeros([batch,cluster_num], theano.config.floatX)
                S[:,D] = 1
                win_counts[D] += 1 
                h = update_rate[D,:].sum(0)[:,None]
                cost = update(ins,S,h)
                
            if verbose:
                print "Avg. centroid distance -- ", cost,"\t EPOCH : ",epoch , " of ", num_epoch
        if self.set_count_activations:
            self.activations += win_counts
            
        if self.set_outlier_unit_det:
            self._update_unit_saliency(win_counts, update_rate, learn_rate)

         # get the data from shared theano variable        
        self.W = W.get_value()

        # Normalize activation counts
        if self.set_count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act

        self.assing_to_units() # final unit assignments

        if self.set_outlier_unit_det:
            self._find_outlier_units()

        if self.set_inunit_outlier_det:
            self._find_inunit_outliers()
            

    def train_batch_theano(self, num_epoch = None, batch_size = None, verbose=True):
        '''
            Theano based batch learning. If you don't define batch size, then all the
            instances are fed for each epoch. 
            
            It is preferred to use batch learning initially then fine tune with 
            stochastic version
            
            In general Theano version is faster if the data is not very small.
        '''
        
        if num_epoch == None:
            num_epoch = 500 * self.X.shape[0]
         
        if batch_size == None:
            batch_size = self.X.shape[0]
        
        # Symmbol variables
        X = T.dmatrix('X')
        WIN = T.dmatrix('WIN')
        alpha = T.dscalar('learn_rate')
        H = T.dmatrix('update_rate')

        # Init weights random
        W = theano.shared(self.W, name='W')
        W_old = W.get_value()

        # Find winner unit
        D = (W**2).sum(axis=1, keepdims=True) + (X**2).sum(axis=1, keepdims=True).T - 2*T.dot(W, X.T)
        BMU = (T.eq(D,D.min(axis=0, keepdims=True))).T
        dist = T.dot(BMU.T, X) - BMU.sum(0)[:, None] * W
        err = D.min(0).sum().norm(1)/X.shape[0] 

        #update = function([X,WIN,alpha],outputs=err,updates=[(W, W + alpha * dist)])
        
        A = T.dot(BMU, H)
        S = A.sum(axis=0)
        update_neigh_no_verbose = function([X, H],outputs=BMU, updates=[(W,  T.where((S[:,None] > 0) ,T.dot(A.T, X), W) / T.where((S > 0), S, 1)[:,None])])
        update_neigh = function([X, H],outputs=[err, BMU], updates=[(W,  T.where((S[:,None] > 0) ,T.dot(A.T, X), W) / T.where((S > 0), S, 1)[:,None])])
        find_bmu = function([X], BMU)

#        if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
#            update_neigh.maker.fgraph.toposort()]):
#            print 'Used the cpu'
#        elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
#            update_neigh.maker.fgraph.toposort()]):
#            print 'Used the gpu'
#        else:
#            print 'ERROR, not able to tell if theano used the cpu or the gpu'
#            print update_neigh.maker.fgraph.toposort()

        U = self.set_params(num_epoch)
        for epoch in range(num_epoch):
            print 'Epoch --- ', epoch
            update_rate = U['H_maps'][epoch]
            learn_rate = U['alphas'][epoch]
            win_counts = np.zeros((self.num_units))
            for i in range(0, self.X.shape[0], batch_size):
                batch_data = self.X[i:i+batch_size, :]
                #temp = find_bmu(batch_data)
                if verbose and epoch % 5 == 0:
                    cost, winners = update_neigh(batch_data, update_rate)
                else:
                    winners = update_neigh_no_verbose(batch_data, update_rate)
                win_counts =+ winners.sum(axis=0)
                ## Normalization is not imperative unless given input instances are normalized
                # self.W = self.W / np.linalg.norm(self.W)
            
                
            if verbose and epoch % 5 == 0:
                print "Avg. centroid distance -- ", cost,"\t EPOCH : ", epoch, " of ", num_epoch
                
            if self.set_count_activations:
                self.activations += win_counts
            
            if self.set_outlier_unit_det:
                self._update_unit_saliency(win_counts, update_rate, learn_rate)

        # get the data from shared theano variable        
        self.W = W.get_value()

        # Normalize activation counts
        if self.set_count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act
        
        self.assing_to_units() # final unit assignments

        if self.set_outlier_unit_det:
            self._find_outlier_units()
        
        if self.set_inunit_outlier_det:
            self._find_inunit_outliers()


    def train_batch(self, num_epoch = None, batch_size = None, verbose=True):
        
        '''
            Numpy version of batch learning
        '''

        if num_epoch == None:
            num_epoch = 500 * self.num_units # Kohonen's suggestion
        
        if batch_size ==  None:
            batch_size = self.X.shape[0]
        
        print 'Learning ... '
        U = self.set_params(num_epoch)
        X2 = (self.X**2).sum(1)[:, None]
        for epoch in range(num_epoch):
            print 'Epoch --- ', epoch
            update_rate = U['H_maps'][epoch]
            learn_rate = U['alphas'][epoch]
            # randomize batch order
            shuffle_indices = np.random.permutation(self.X.shape[0])
            win_counts = np.zeros((self.num_units))
            for batch_indices in  np.array_split(shuffle_indices, self.X.shape[0]/batch_size):
                batch_data = self.X[batch_indices,:]
                D = self._euq_dist(X2[batch_indices,:], batch_data)
                BMU = (D==D.min(0)[None,:]).astype("float32").T
                
                win_counts += BMU.sum(axis=0)
                #print win_counts
                
                if self.set_count_activations:
                    self.activations += win_counts
                
                # batch learning
                A = np.dot(BMU, update_rate)
                S = A.sum(0)
                non_zeros = S.nonzero()[0]
                self.W[non_zeros, ...] =  np.dot(A[:,non_zeros].T, batch_data) / S[non_zeros][..., None]
                
                # normalize weight vector
                ## Normalization is not imperative unless given input instances are normalized
                # self.W = self.W / np.linalg.norm(self.W)
                #self.W = self.W / np.linalg.norm(self.W)
        
            if self.set_outlier_unit_det:
                self._update_unit_saliency(win_counts, update_rate, learn_rate)
                    
            if verbose and ((epoch % 1) == 0):
               self._print_cost(X2, epoch, num_epoch)

        # Normalize activation counts
        if self.set_count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act
        
        self.assing_to_units() # final unit assignments

        if self.set_outlier_unit_det:
            self._find_outlier_units()
        
        if self.set_inunit_outlier_det:
            self._find_inunit_outliers()
            
      
    # Uses the Chessboard distance
    # Find the neighbooring units to given unit
    vis_neigh = lambda neigh_map, indx : neigh_map[indx].reshape((self.height, self.width))
    def find_neighbors(self, unit_id, radius):
        neighbors = np.zeros((1,self.num_units))      
        test_neig = np.zeros((self.height, self.width))
        unit_x, unit_y = self.unit_cords(unit_id) 
        
        min_y = max(int(unit_y - radius), 0)
        max_y = min(int(unit_y + radius), self.height-1)
        min_x = max(int(unit_x - radius), 0)
        max_x = min(int(unit_x + radius), self.width-1)
        for y in range(min_y, max_y+1,1):
            for x in range(min_x, max_x+1,1):
                dist = abs(y-unit_y) + abs(x-unit_x)
                neighbors[0, x + ( y * self.width )] = dist
                test_neig[y,x] = dist
        return neighbors
    
    # find BMUs and between-distances for given set of instances
    def best_match(self, X):
        if len(X.shape) == 1:

            X = X.reshape((1,2))
        X2 = (self.X**2).sum(1)[:, None]
        D = -2*np.dot(self.W, X.T)[None,:] + (self.W**2).sum(1)[:, None] + X2.T
        BMU = (D==D.min(0)[None,:]).astype("float32").T
        return BMU, D
    
    # structure the unit weight to be shown at U map
    def som_map(self):
        print('Som mapping is being computed...')
        sqrt_weigths = np.reshape(self.W,(self.height, self.width, self.data_dim))
        um = np.zeros((sqrt_weigths.shape[0],sqrt_weigths.shape[1]))
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1,it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1,it.multi_index[1]+2):
                    if ii >= 0 and ii < sqrt_weigths.shape[0] and jj >= 0 and jj < sqrt_weigths.shape[1]:
                        um[it.multi_index] += np.linalg.norm(sqrt_weigths[ii,jj,:]-sqrt_weigths[it.multi_index])
            it.iternext()
        um = um/um.max()
        print("Mapping finished...!")
        return um  
        

    # set the ratio of width and height of the map by the 
    # ratio between largest 2 eigenvalues, computed from data
    def _estimate_map_shape(self):
        #num_instances = self.X.shape[0]
        u,s,v = np.linalg.svd(self.X ,full_matrices = False)
        s_sorted = np.sort(s)[::-1]
        ratio = s_sorted[0] / s_sorted[1]
        self.height = int(min(self.num_units, np.ceil(np.sqrt(self.num_units / ratio))))
        self.width = int(np.ceil(self.num_units / self.height))
        # self.height = int(np.round(np.sqrt(num_instances)))
        # self.width = int(np.round(num_instances / self.height))
        print 'Estimated map size is -> height = ', self.height, ' width = ',self.width 

    # assign instances to matching BMUs
    def assing_to_units(self, X=None):
        if X == None:
            X2 = (self.X**2).sum(1)[:, None]
            D = -2*np.dot(self.W, self.X.T) + (self.W**2).sum(1)[:, None] + X2.T

            self.ins_unit_assign = D.argmin(axis=0)
            self.ins_unit_dist = D[self.ins_unit_assign, np.arange(self.X.shape[0])]
        else:
            X2 = (X**2).sum(1)[:, None]
            D = -2*np.dot(self.W, X.T) + (self.W**2).sum(1)[:, None] + X2.T
            ins_unit_assign = D.argmin(axis=0)
            ins_unit_dist = D[ins_unit_assign, np.arange(X.shape[0])]
            return ins_unit_assign , ins_unit_dist

        
    def find_units_coherence(self):
        
        '''
            Find individually coherence of each unit by looking to avg. distance
            between unit weight and the assigned instances
        '''
        
        self.unit_coher = np.zeros((self.num_units))
        for i in np.unique(self.ins_unit_assign):
            indices = np.where(self.ins_unit_assign == i)
            self.unit_coher[i] = np.sum(self.ins_unit_dist[indices]) / indices[0].size

    # return BMU, BMU distance, saliency by already trained params
    def process_new_data(self, X):
        BMU,dist = self.assing_to_units(X)

        # find outlier instanes in outlier units
        ins_saliency= np.ones((X.shape[0]), dtype=bool)
        outlier_units = np.where(self.unit_saliency == False)[0]
        for i in outlier_units:
            ins_saliency[np.where(BMU == i)] = False

        # find salient unit outliers
        for i in np.unique(BMU):
            indices = np.where(BMU == i)[0]
            unit_thresh = scipy.stats.scoreatpercentile(dist[indices], self.inunit_outlier_thresh)
            outlier_insts = indices[dist[indices] > unit_thresh]
            ins_saliency[outlier_insts] = False;

        return BMU, dist, ins_saliency



    def _update_unit_saliency(self, win_counts, update_rate, learn_rate):
        
        '''
            It is called after each epoch of the learning. It compute the 
            unit saliencies with the paper formula. At the end, those values
            defines the outlier and salient units
        '''
        
        excitations = (update_rate * win_counts).sum(axis=0) / learn_rate
        excitations = excitations / excitations.sum()
        single_excitations = win_counts * learn_rate
        single_excitations = single_excitations / single_excitations.sum()
        self.unit_saliency_coeffs += excitations + single_excitations
        
    def _find_outlier_units(self):
        
        '''
            After we compute unit saliencies, this function detects the outlier
            units by the paper heuristic
        '''
        
        # find outlier units
        self.unit_saliency_coeffs /= self.unit_saliency_coeffs.sum()
        self.unit_saliency = self.unit_saliency_coeffs > self.outlier_unit_thresh/self.num_units

        # sign outlier instances
        self.inst_saliency = np.ones((self.X.shape[0]), dtype=bool)
        outlier_units = np.where(self.unit_saliency == False)[0]
        for i in outlier_units:
            self.inst_saliency[np.where(self.ins_unit_assign == i)] = False
    
    def _find_inunit_outliers(self):

        '''
            Find the poor instances at the salient units. It uses an upper whisker
            assigned to the distances of the unit weight to unit instances. given the threshold,
            outside of the whisker is detedted as outlier.
        '''        
        
        # #remove outlier units
#        int_units = np.array(range(self.num_units))
#        if self.unit_saliency.size > 0 and self.set_inunit_outlier_det:
#            int_units = int_units[self.unit_saliency]
        if self.inst_saliency.size == 0:
            self.inst_saliency = np.ones((self.X.shape[0]), dtype=bool)
            
        for i in np.unique(self.ins_unit_assign):
            indices = np.where(self.ins_unit_assign == i)[0]
            unit_thresh = scipy.stats.scoreatpercentile(self.ins_unit_dist[indices], self.inunit_outlier_thresh)
            outlier_insts = indices[self.ins_unit_dist[indices] > unit_thresh]
            self.inst_saliency[outlier_insts] = False;

    # Returns indices of salient instances
    def salient_inst_index(self):
        return np.where(self.inst_saliency == True)[0]

    def salient_unit_index(self):
        return np.where(self.unit_saliency == True)[0]  
        
    def salient_insts(self):
        return self.X[np.where(self.inst_saliency == True)]

    def salient_units(self):
        return self.W[np.where(self.unit_saliency == True)]

    ## Returns instance to unit mapping. First row is instances.
    def inst_to_unit_mapping(self):
        return np.concatenate((np.arange(self.X.shape[0])[None,:], self.ins_unit_assign[None, :]))

    def salient_inst_to_unit_mapping(self):
        mapping = self.inst_to_unit_mapping()
        

    def _norm_data(self, X = None):
        
        '''
            Take the norm of the given data matrix and save std and mean 
            for future purposes
        '''
        
        if X == None:
            self.data_mean =  self.X.mean(axis=0)
            self.data_std  =  self.X.std(axis=0, ddof=1)
            self.X = (self.X - self.data_mean) / (self.data_std  + EPS)
        else:
            data_mean =  X.mean(axis=0)
            data_std  =  X.std(axis=0, ddof=1)
            X = (X - data_mean) / data_std
            return X, data_mean, data_std
            
   
'''
DEMO CODE
'''
if __name__ == "__main__":
    from sklearn import datasets
    import time

    data = datasets.load_digits().data

    som = SOM(DATA = data, alpha_max=0.05, num_units=100, height = 10, width = 10)
    #som.train_batch(100)
    #start = time.time()
    #som.train_stoch_theano(10)
    som.train_batch_theano(num_epoch=100)
    #som.train_stoch(10)
    #clusters = som.ins_unit_assign
    #print clusters
    #stop = time.time()
    #
    print som.unit_saliency

    #som_plot_scatter(som.W, som.X, som.activations)    
    #som_plot_outlier_scatter(som.W, som.X, som.unit_saliency, som.inst_saliency, som.activations)
    #som_mapping = som.som_map()
    #som_plot_mapping(som_mapping)
    print "Demo finished!"
    #print "Pass time : ", stop - start
   
Rectifying Self Organizing Map (RSOM)
===============================

For More Detail Visit: <a href='http://www.erengolge.com/pub_sites/concept_map.html'> Project Page </a>.

<div class="image" align='center'>
				<img src="http://i57.tinypic.com/2nv7yon.gif" alt="Image and video hosting by TinyPic" width="50%" height="50%"/>
				<div align="bottom"><b>CMAP in runtime</b>. It shows the unit updates from iteration 1 to saturation where the units are precisely defined.</div>
				</div>


Implemented and designed by <a href='http://www.erengolge.com'>Eren Golge</a> for the work <b>"Gölge, E., & Duygulu, P.. ConceptMap:Mining noisy web data for concept learning , The European Conference on Computer Vision (ECCV) 2014." </b>

RSOM is an algorithm as an extension of well-known Self Organizing Map (SOM). It mimics SOM clustering and additionally detects outliers in the given dataset in the cluster level or instance level.

Example call below with different commented call alternative.

There are two different implementations for training. <b>Scipy</b> implementation is designed for small scale problems and <b>Theano</b> version is for large scale problems which GPU utilization might help.

You can use this code for only SOM clustering as well. Up to my knowledge, it is the only SOM library designed for large data-sets in Python.

<pre>
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
</pre>

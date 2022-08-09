

import os
import igraph as ig
import numpy as np
import os
import codecs
import random
from sklearn.metrics import *
from collections import *
from copy import deepcopy
import time
import math

FOLDER_GRAPH = 'data-time'
import sys

def generate_graphs(N, tau1, tau2, mu, k, maxk, minc=None, maxc=None,count=1):
    """ generate count graphs and save them to data\filename-****.xml """
    # https://sites.google.com/site/santofortunato/inthepress2
    '''
    -N number of vertices
    -k		average degree
    -maxk		maximum degree
    -mu		mixing parameter
    -t1		minus exponent for the degree sequence
    -t2		minus exponent for the community size distribution
    -minc		minimum for the community sizes
    -maxc		maximum for the community sizes
    -on		number of overlapping nodes
    -om		number of memberships of the overlapping nodes
    -C              [average clustering coefficient]
    '''
    graphs = []
    counter = 0

    while counter < count:

        time.sleep(1.0)

        os.system("rm network.dat community.dat")

        if minc is None and maxc is None:
            s = os.system("./benchmark -N {0} -k {1} -maxk {2} -mu {3}".format(N, k, maxk, mu))
        else:
            s = os.system(
                "./benchmark -N {0} -k {1} -maxk {2} -mu {3} -minc {4} -maxc {5}".format(N, k, maxk, mu, minc,
                                                                                   maxc))
        assert s == 0

        g_LFR, file, file_data,  = None, None, None

        g_LFR = ig.Graph.Read_Edgelist(r'network.dat', directed=False)
        g_LFR.delete_vertices(0)
        g_LFR.simplify()

        file = codecs.open(r'community.dat')
        file_data = np.loadtxt(file, usecols=(0, 1), dtype=int)

        clusters = file_data[:, 1]
        g_LFR.vs['cluster'] = clusters
        #print("CLUSTERS",g_LFR.clusters())
        print(set(g_LFR.vs['cluster']))

        #if len(g_LFR.clusters()) != 1 and len(g_LFR.community_leading_eigenvector()) < 1:
        try:
            if len(set(g_LFR.vs['cluster'])) != 1 and len(g_LFR.community_leading_eigenvector()) >= 1:
                graphs.append(g_LFR.copy())
                g_LFR = None
                counter += 1
        except:
            pass

    return graphs

def write_graphs(graphs,  filename='graph'):



    for counter, g in enumerate(graphs):
        g.write_graphml("./"+FOLDER_GRAPH +"/{}-{:03d}.xml".format(filename, counter))
        print("Graph written to ./"+FOLDER_GRAPH +"/{}-{:03d}.xml".format(filename, counter))



############# GENERATE GRAPHS #############

#n_verts = [100, 316, 1000, 3162, 10000, 31622, 316227, 1000000]
n_verts= [177, 316, 562, 1000, 1778, 3162, 5623, 10000, 17782, 31622, 56234, 100000] #, 177827, 316227, 562341, 1000000]
n_samples = [100,100,100,100,100,100,100,50,50,25,10,10]
#graphs = generate_graphs(N=500, tau1=2, tau2=1, m.5, k=15, maxk=50, minc=50, maxc=100, count=100)
#write_graphs(graphs, filename='pet-'+str(500)+'-mu-'+str(0.5))
mt_ind = int(sys.argv[1])
n_s = n_samples[mt_ind]
n_v = n_verts[mt_ind]
mu = 0.25
k = 15



graphs = generate_graphs(N=n_v, tau1=2, tau2=1, mu=mu, k=k, maxk=3 * k, minc=n_v // 20, maxc=n_v // 10, count=n_s)
write_graphs(graphs, filename='graph-' + str(n_v) + '-mu-' + str(mu))

"""


for n_s, n_v in zip(n_samples,n_verts):
    mu = 0.25
    k = 15
    graphs = generate_graphs(N=n_v, tau1=2, tau2=1, mu=mu, k=k, maxk=3*k, minc=n_v//20, maxc=n_v//10, count=n_s)
    write_graphs(graphs, filename='graph-'+str(n_v)+'-mu-'+str(mu))
"""

os.system("touch finished.txt")
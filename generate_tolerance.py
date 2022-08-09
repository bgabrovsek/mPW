

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

FOLDER_GRAPH = 'data-tol'

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
        try:
            os.system("rm network.dat community.dat")
        except:
            pass
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
        g.write_graphml("./"+FOLDER_GRAPH+"/{}-{:03d}.xml".format(filename, counter))
        print("Graph written to ./"+FOLDER_GRAPH+"/{}-{:03d}.xml".format(filename, counter))

nn = 500
mu = 0.1
graphs = generate_graphs(N=nn, tau1=2, tau2=1, mu=mu, k=15, maxk=30, minc=25, maxc=50, count=100)
write_graphs(graphs, filename='graph-'+str(nn)+'-mu-'+str(mu))

os.system("touch finished_500.txt")
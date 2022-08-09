


import os

import igraph
import igraph as ig
import numpy as np
import os
import random
from sklearn.metrics import *
from collections import *
import time
import math



def wr(filename, s = None):
    file_object = open(filename, 'w' if s is None else 'a')
    if s is not None:
        if s != "\n":
            file_object.write(str(s)+", ")
        else:
            file_object.write("\n")

    file_object.close()


def conductance(g, c):
    """Evaluates the conductance of the clustering solution c on a given graph g."""
    n = len(g.vs)
    m = len(g.es)

    ASSc = {}
    AS = {}

    for col in set(c):
        ASSc[col] = 0
        AS[col] = 0

    for i in range(n):
        current_clr = c[i]
        for j in g.neighbors(i):
            if c[i] == c[j]:
                AS[current_clr] += 1 / 2
            else:
                ASSc[current_clr] += 1
                AS[current_clr] += 1

    phi_S = {}

    for col in set(c):
        if (min(AS[col], m - AS[col] + ASSc[col]) == 0):
            phi_S[col] = 0
        else:
            phi_S[col] = ASSc[col] / min(AS[col], m - AS[col] + ASSc[col])

    phi = 0

    for col in set(c):
        phi -= phi_S[col]

    phi /= len(set(c))
    phi += 1

    intra_cluster_phi = min(phi_S.values())
    inter_cluster_phi = 1 - max(phi_S.values())

    return phi, intra_cluster_phi, inter_cluster_phi


def coverage(g, c):
    """Evaluates the coverage of the clustering solution c on a given graph g."""
    n = len(g.vs)
    A_delta = 0
    A = 0

    for i in range(n):
        for j in g.neighbors(i):
            A += 1
            if c[i] == c[j]:
                A_delta += 1

    return A_delta / A


def load_graphs(folder=None, filename=None, n = 1000):
    print("LOADING GRAPHS.", folder, filename, n)
    """ Load graphd from files data\filename-****.xml """
    graphs = []
    for counter in range(0,n):
        #try:
        #print("./data/{}-{:03d}.xml".format(filename, counter))
        #if os.name == 'nt':
        #    g = ig.Graph.Read_GraphML("{}\\{}-{:03d}.xml".format(folder,filename, counter))
        #else:
        fn = "./{}/{}-{:03d}.xml".format(folder,filename, counter)
        #print(fn)
        g = ig.Graph.Read_GraphML(fn)

        g.vs['cluster'] = [int(x) for x in g.vs['cluster']]  # clusters to int

        #print("loaded", g)

        graphs.append(g)
    #except:
        #    break
    return graphs
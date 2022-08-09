# -*- coding: utf-8 -*-
"""
@author: Barbara Ikica
"""

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
import pandas as pd

def plot_clusters(g, c):
    """
    Draws a given graph g with vertex colours corresponding to clusters c and
    displays the corresponding sizes of the clusters.
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    g : a graph
    c : a list of vertex colours (clusters)
    ---------------------------------------------------------------------------
    """

    if type(c) == dict:
        c = list(c.values())

    g.vs['color'] = c
    g.vs['label'] = c

    palette = ig.ClusterColoringPalette(len(g.vs))

    df = pd.DataFrame(columns=['Frequency'])
    df.index.name = 'Colour'

    for i in set(c):
        df.loc[int(i)] = [c.count(i)]

    display(df)

    visual_style = {}
    visual_style['vertex_color'] = [palette[col] for col in g.vs['color']]
    visual_style['vertex_label'] = [col for col in g.vs['color']]
    visual_style['vertex_frame_width'] = 0
    visual_style['bbox'] = (1000, 1000)
    visual_style['margin'] = 10

    return ig.plot(g, **visual_style)


def plot_results(badVert, badEdges):
    """
    Plots the number of bad vertices and the number of bad edges over time.
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    badVert : a list of the number of bad vertices over time
    badEdges : a list of of the number of bad edges over time
    ---------------------------------------------------------------------------
    """

    plt.title('# bad vertices');
    plt.plot(badVert);
    plt.show();

    plt.title('# bad edges');
    plt.plot(badEdges);
    plt.show();


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


def association_matrix(c):
    """Computes the association matrix corresponding to the clustering solution c.
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    c : clustering solution
    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    C : the association matrix with 1 iff i and j are in the same cluster
    modules : a dictionary of cluster memberships
    ---------------------------------------------------------------------------
    """
    if type(c) == dict:
        c = list(c.values())

    n = len(c)

    C = np.zeros((n, n))  # association matrix

    outk = len(set(c))  # final number of clusters
    # list of vertices corresponding to each of the clusters
    modules = dict.fromkeys(range(outk))

    for j in list(set(c)):
        cj = [ind for ind, val in enumerate(c) if val == j]
        modules[j] = cj
        C[np.ix_(cj, cj)] = 1

    return C, modules

import os
def load_graphs(filename=None, n = 1000):
    """ Load graphd from files data\filename-****.xml """
    graphs = []
    for counter in range(2,2+n):
        try:
            #print("./data/{}-{:03d}.xml".format(filename, counter))
            if os.name == 'nt':
                g = ig.Graph.Read_GraphML("data-500\\{}-{:03d}.xml".format(filename, counter))
            else:
                g = ig.Graph.Read_GraphML("./data-500/{}-{:03d}.xml".format(filename, counter))

                g.vs['cluster'] = [int(x) for x in g.vs['cluster']]  # clusters to int
            graphs.append(g)
        except:
            break
    return graphs

def compare_clustering_woTN(c1, c2):
    C1, modules1 = association_matrix(c1)
    C2, modules2 = association_matrix(c2)

    v1 = C1[np.triu_indices_from(C1, 1)]
    v2 = C2[np.triu_indices_from(C2, 1)]

    ind = np.where(np.logical_not((np.vstack((v1, v2)) == 0).all(axis=0)))
    v1 = v1[ind]
    v2 = v2[ind]

    return v1, v2


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
            s = os.system("a -N {0} -k {1} -maxk {2} -mu {3}".format(N, k, maxk, mu))
        else:
            s = os.system(
                "a -N {0} -k {1} -maxk {2} -mu {3} -minc {4} -maxc {5}".format(N, k, maxk, mu, minc,
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


#graphs = load_graphs(filename='gra-' + str(nn) + '-mu-'+str(mu),n=50)

nn = 500
mu = 0.1
graphs = generate_graphs(N=nn, tau1=2, tau2=1, mu=mu, k=15, maxk=30, minc=50, maxc=100, count=1)

g = graphs[0]
p = plot_clusters(g, g.vs['cluster'])
p.show()
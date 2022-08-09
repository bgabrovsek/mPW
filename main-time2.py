

import os

import igraph
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
from threading import Thread
from sklearn.semi_supervised import LabelPropagation

FOLDER_GRAPH = 'data-time'
import sys
my_tolerance = 100.0

mpw_time1 = 'cmpv-time1-'+str(my_tolerance)+'.csv'
mpw_time2 = 'cmpv-time2-'+str(my_tolerance)+'.csv'
mpw_time3 = 'cmpv-time3-'+str(my_tolerance)+'.csv'
mpw_cluster = 'cmpv-clusters-'+str(my_tolerance)+'.csv'

# MULTITASK

def variance_first_time(seznam):
    return np.var(seznam), np.sum(seznam), np.sum(np.square(seznam))


def update_variance(seznam, vsota, vsota_kvadratov, indeks, nova_vrednost):
    n = len(seznam)
    stara_vrednost = seznam[indeks]
    seznam[indeks] = nova_vrednost
    vsota += (nova_vrednost - stara_vrednost)
    vsota_kvadratov += (nova_vrednost ** 2 - stara_vrednost ** 2)
    varianca = (vsota_kvadratov - vsota**2 / n) / n
    #print(seznam)
    return varianca, vsota, vsota_kvadratov


def wr(filename, s = None):
    file_object = open(filename, 'w' if s is None else 'a')
    if s is not None:
        if s != "\n":
            file_object.write(str(s)+", ")
        else:
            file_object.write("\n")

    file_object.close()

def mpw_single(g, tolerance, result, index):
    #wr('tol.csv')

    w = 6.0
    n = g.vcount()  # number of vertices
    win_size = 500
    #max_iterations = n * n * n # int(math.sqrt(n))
    coloring = np.random.permutation(n) # number of colors = number if vertices

    adj_vertices = [g.neighbors(v) for v in range(n)]  # adjacent vertices
    bad_vertices = set([v for v in range(n) if np.any(coloring[adj_vertices[v]] != coloring[v])])  # bad vertices
    # number of bad edges sliding windowddd
    number_bad_edges = np.zeros((win_size,))
    number_bad_edges[::2] = 500.0 # variance = 25.0, something with big variance
    number_bad_edges[1::2] = -500.0  #
    step = 0
    number_bad_edges[step % win_size] = sum(
        [coloring[e.tuple[0]] != coloring[e.tuple[1]] for e in g.es])  # add num. bad edges
    # print("coloring", coloring, "\nadj verts", adj_vertices,"\n# bad edges", number_bad_edges)
    #print("tolerance", tolerance)
    # main loop

    vart, vsota, vsota_sq = variance_first_time(number_bad_edges)

    #print(vart, tolerance, number_bad_edges[:30])

    while (number_bad_edges[step % win_size] > 0) and (vart >= tolerance): #and (
            #step < max_iterations):

        vertex = random.choice(list(bad_vertices))
        adj_colors = Counter(coloring[adj_vertices[vertex]])
        old_color = coloring[vertex]
        probs = w ** np.array(list(adj_colors.values()))
        new_color = random.choices(list(adj_colors.keys()), probs, k=1)[0]
        coloring[vertex] = new_color
        # update number of bad edges
        nbe = number_bad_edges[step % win_size]
        nbe -= sum(coloring[v] != old_color for v in adj_vertices[vertex])  # number of old bad edges
        nbe += sum(coloring[v] != new_color for v in adj_vertices[vertex])  # number of new bad edges
        step += 1


        # print("vertex", vertex, "color", old_color, "->", new_color)

        # update list of bad vertices
        bad_vertices -= set(adj_vertices[vertex]) | {vertex}  # remove all adjacent vertices from set
        bad_vertices |= set(
            v for v in (adj_vertices[vertex]) if any(coloring[adj_vertices[v]] != coloring[v]))  # add new bad vertices
        if any(coloring[adj_vertices[vertex]] != coloring[vertex]):
            bad_vertices |= {vertex}

        vart, vsota, vsota_sq = update_variance(number_bad_edges, vsota, vsota_sq, step % win_size, nbe)
        #number_bad_edges[step % win_size] = nbe  # update sliding window
        #vart = np.var(number_bad_edges)
        #wr('tol.csv', str(step) + ", " + str(vart)+"\n")

    result[index] = coloring






def mpw_seq(graphs, tol, repeats=1):
    """ modified pw algorithm, multiple graphs run multiple times"""

    if not isinstance(graphs, list):
        graphs = [graphs]

    colorings = []
    for index, g in enumerate(graphs):
        print(".", end="", flush=True)
        results = [None] * repeats
        for i in range(len(results)):
            mpw_single(g, tol, results, i)
        colorings.append(results)

    return colorings




def mpw_seq_lite(graphs, tol, repeats):


    """ modified pw algorithm, multiple graphs run multiple times"""

    results = [None]
    n = graphs[0].vcount()
    for index, g in enumerate(graphs):
        wr(mpw_time1, n)
        wr(mpw_time2, n)
        wr(mpw_time3, n)
        wr(mpw_cluster, n)

        wr(mpw_time1, str(index))
        wr(mpw_time2, str(index))
        wr(mpw_time3, str(index))
        wr(mpw_cluster, str(index))

        for i in range(repeats):
            start1 = time.time()
            start2 = time.process_time()
            start3 = time.process_time_ns()

            mpw_single(g, tol, results, 0)

            time1 = time.time() - start1
            time2 = time.process_time() - start2
            time3 = time.process_time_ns() - start3
            wr(mpw_time1, str(time1))
            wr(mpw_time2, str(time2))
            wr(mpw_time3, str(time2))
            wr(mpw_cluster, str(len(set(results[0]))))

        wr(mpw_time1, "\n")
        wr(mpw_time2, "\n")
        wr(mpw_time3, "\n")
        wr(mpw_cluster, "\n")

        print(index, time2, str(len(set(results[0]))))



def load_graphs(filename=None, n = None):
    """ Load graphd from files data\filename-****.xml """
    graphs = []
    for counter in range(n):
        try:
            #print("./data/{}-{:03d}.xml".format(filename, counter))
            if os.name == 'nt':
                g = ig.Graph.Read_GraphML("data-time\\{}-{:03d}.xml".format(filename, counter))
            else:
                g = ig.Graph.Read_GraphML("./data-time/{}-{:03d}.xml".format(filename, counter))

                g.vs['cluster'] = [int(x) for x in g.vs['cluster']]  # clusters to int
            graphs.append(g)
        except:
            break
    return graphs



######### COMPUTE NMS #########

wr(mpw_time1)
wr(mpw_time2)
wr(mpw_time3)
wr(mpw_cluster)

#nn = 1000
#mu = 0.5
#graphs = load_graphs(filename='exp-' + str(nn) + '-mu-'+str(mu),n=10)

#mpw_seq_lite(graphs, 0.009, repeats=10)
#exit()
n_verts= [177, 316, 562, 1000, 1778, 3162, 5623, 10000, 17782, 31622, 56234, 100000] #, 177827, 316227, 562341, 1000000]
n_samples = [100,100,100,100,100,100,100,50,50,25,10,10]
n_poskusov = 5

mt_ind = int(sys.argv[1])
n_s = n_samples[mt_ind]//n_poskusov
n_v = n_verts[mt_ind]
mu = 0.25
print("Loading", "samp", n_s, "n", n_v, "mu", mu, "tol", my_tolerance)

graphs = load_graphs(filename='graph-' + str(n_v) + '-mu-'+str(mu),n=n_s)
mpw_seq_lite(graphs, my_tolerance, repeats=n_poskusov)


    #lp_seq_lite(graphs, 0.001, repeats=ns)



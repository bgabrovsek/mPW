

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

# MULTITASK

# single thread
def mpw_single(g, tolerance, result, index):
    w = 6.0
    n = g.vcount()  # number of vertices
    win_size = n
    max_iterations = n * int(math.sqrt(n))
    coloring = np.random.permutation(n) # number of colors = number if vertices

    adj_vertices = [g.neighbors(v) for v in range(n)]  # adjacent vertices
    bad_vertices = set([v for v in range(n) if np.any(coloring[adj_vertices[v]] != coloring[v])])  # bad vertices
    # number of bad edges sliding window
    number_bad_edges = np.arange(win_size) * 100  # something with big variance
    step = 0
    number_bad_edges[step % win_size] = sum(
        [coloring[e.tuple[0]] != coloring[e.tuple[1]] for e in g.es])  # add num. bad edges
    # print("coloring", coloring, "\nadj verts", adj_vertices,"\n# bad edges", number_bad_edges)

    # main loop
    while (number_bad_edges[step % win_size] > 0) and (np.var(number_bad_edges) >= tolerance) and (
            step < max_iterations):

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
        number_bad_edges[step % win_size] = nbe  # update sliding window

        # print("vertex", vertex, "color", old_color, "->", new_color)

        # update list of bad vertices
        bad_vertices -= set(adj_vertices[vertex]) | {vertex}  # remove all adjacent vertices from set
        bad_vertices |= set(
            v for v in (adj_vertices[vertex]) if any(coloring[adj_vertices[v]] != coloring[v]))  # add new bad vertices
        if any(coloring[adj_vertices[vertex]] != coloring[vertex]):
            bad_vertices |= {vertex}
    result[index] = coloring


def mpw_multi(graphs, tol, repeats=1):
    """ modified pw algorithm, multiple graphs run multiple times"""

    if not isinstance(graphs, list):
        graphs = [graphs]

    colorings = []
    for g in graphs:

        #print(".", end="", flush=True)

        threads = [None] * repeats
        results = [None] * repeats
        print("creating threads")
        for i in range(len(threads)):
            threads[i] = Thread(target=mpw_single, args=(g.copy(), tol, results, i)) #g, tolerance, result, index
            threads[i].start()
        print("ran")
        for i in range(len(threads)):
            threads[i].join()
        print("joined.")
        colorings.append(results)
    print()
    return colorings

def w(filename, s = None, newline = False):
    file_object = open(filename, 'w' if s is None else 'a')
    file_object.write(str(s) + ("\n" if newline else ", "))
    file_object.close()


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




def mpw_seq_lite(graphs, tol, repeats=1):
    """ modified pw algorithm, multiple graphs run multiple times"""
    results = [None]
    n = graphs[0].vcount()
    for index, g in enumerate(graphs):
        w('mpw.csv', n)
        w('mpw.csv', 'G'+str(index))
        for i in range(repeats):
            start = time.time()
            mpw_single(g, tol, results, 0)
            w('mpw.csv', time.time()-start)

def lp_seq_lite(graphs, tol, repeats=1):
    """ modified pw algorithm, multiple graphs run multiple times"""
    results = [None]
    n = graphs[0].vcount()
    for index, g in enumerate(graphs):
        w('lp.csv', n)
        w('lp.csv', 'G'+str(index))
        for i in range(repeats):
            start = time.time()
            c = g.community_label_propagation()
            w('lp.csv', time.time()-start)




def load_graphs(filename=None, n = 1000):
    """ Load graphd from files data\filename-****.xml """
    graphs = []
    for counter in range(n):
        try:
            #print("./data/{}-{:03d}.xml".format(filename, counter))
            if os.name == 'nt':
                g = ig.Graph.Read_GraphML("data2\\{}-{:03d}.xml".format(filename, counter))
            else:
                g = ig.Graph.Read_GraphML("./data2/{}-{:03d}.xml".format(filename, counter))

                g.vs['cluster'] = [int(x) for x in g.vs['cluster']]  # clusters to int
            graphs.append(g)
        except:
            break
    return graphs

############# GENERATE GRAPHS #############
"""
for mu in [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    graphs = generate_graphs(N=1000, tau1=2, tau2=1, mu=mu, k=15, maxk=100, minc=50, maxc=100, count=10)
    write_graphs(graphs, filename='graph-1000-mu-'+str(mu))
#exit()
"""

def compute_nmi(graphs, colorings):
    results = []
    for g, cs in zip(graphs, colorings):
        nmis = [normalized_mutual_info_score(g.vs['cluster'], c) for c in cs]
        results.append(max(nmis))
    return results

######### COMPUTE NMS #########
nmis = []
nmis2 = []

mus = [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
tols = [0.0001, 0.0001, 0.0002, 0.0002, 0.009, 0.01, 0.03, 0.03, 0.03]

w('mpw.csv')
w('lp.csv')

# loop through mu's
for mu, tol in zip(mus, tols):


    print("mu",mu,"tol",tol, end=" ", flush=True)
    graphs = load_graphs(filename='graph-1000-mu-'+str(mu),n=3)

    print("loaded.")

    start_time_mpw = time.time()
    colorings = mpw_seq(graphs, tol, repeats=2)
    time_mpw = time.time() - start_time_mpw


    start_time_lp = time.time()


    colorings =  LabelPropagation()  #  mpw_seq(graphs, tol, repeats=2)
    time_lp = time.time() - start_time_lp


    NMI = compute_nmi(graphs, colorings)
    print("NMI =",NMI)

    """
    for rep in range(N_rep):

        colorings = mpw(graphs, tolerance = tol)
        #print([set(c) for c in colorings])

        nmi_1 = [normalized_mutual_info_score(g.vs['cluster'], c) for g, c in zip(graphs, colorings)]

        nmi_max = np.maximum(nmi_max, nmi_1)

        #print("mu",mu,"nmi", nmi)
    print("NMI MAX", nmi_max)
    nmis.append(sum(nmi_max)/len(nmi_max))

    #nmis2.append(sum(nmi2)/len(nmi2))
    """

    # threading: 155.7824990749359 seconds
    # seq: 60.25725841522217 seconds

print(nmis)
#print(nmis2)

######### COMPUTE NMS #########

# RESULT NMIS
#[0.2958500245497557, 0.28163344755619935, 0.2894475769118044, 0.278824151692989, 0.2734609944260655, 0.27181600586071947, 0.2743177980399902, 0.27046159704584316, 0.27448966653263174]
#[0.27744912079543566, 0.2839001565443249, 0.27287812393681016, 0.2719271528595221, 0.27717653306893064, 0.2732965519970335, 0.26951583244257726, 0.2734436411465667, 0.2676654906123931]



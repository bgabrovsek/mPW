

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
from grph import *
from networkx.generators.community import LFR_benchmark_graph

FOLDER_GRAPH = 'data-tol'
CSV = "mpw-tolerance.csv"


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


def mpw_single(g, tolerance, result, index):
    #wr('tol.csv')

    w = 6.0
    n = g.vcount()  # number of vertices
    win_size = n
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

def quality(g, c):
    """
    :return: quality indices of coloring c
    """
    result = dict()
    average = 'micro' # 'micro', 'macro', 'samples','weighted', 'binary', None
    result['NMI'] = normalized_mutual_info_score(g.vs['cluster'], c)
    result['AMI'] = adjusted_mutual_info_score(g.vs['cluster'], c)
    result['ARI'] = adjusted_rand_score(g.vs['cluster'], c)
    result['phi'] = conductance(g, c)[0]
    result['gamma'] = coverage(g, c)
    result['Q'] = g.modularity(c)
    result['F1'] = f1_score(g.vs['cluster'], c, average = average)
    result['F2'] = fbeta_score(g.vs['cluster'], c, beta=2, average = average)
    result['FM'] = fowlkes_mallows_score(g.vs['cluster'], c)
    result['JI'] = jaccard_score(g.vs['cluster'], c, average = average)
    result['V']  = v_measure_score(g.vs['cluster'], c)
    result['|c|'] = len(set(c))
    result['|g|'] = len(set(g.vs['cluster']))
    return result

def mpw_single_100(g, tolerance):

    q_keys = ['NMI', 'ARI','phi','gamma','Q','F1','F2','FM','JI','V','|c|','|g|']

    best_q = {key:-10000.0 for key in q_keys}

    lcl = 0

    all_time = 0.0

    result = [None]
    for gi in range(100):

        start_time = time.time()

        mpw_single(g, tolerance, result, 0)
        col = result[0]
        all_time += (time.time() - start_time)
        q = quality(g, col)

        best_q = {key: max(q[key], best_q[key]) for key in q_keys  }
        if best_q['NMI'] == q['NMI']:
            lcl = q['|c|']

    best_q['|c|'] = lcl
    best_q['time'] = all_time


    print("\nBEST\n", best_q)
    return best_q





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
        wr(CSV, n)
        wr(CSV, 'G'+str(index))
        for i in range(repeats):
            start = time.time()
            mpw_single(g, tol, results, 0)
            wr(CSV, time.time()-start)

        wr(CSV, "\n")


tolerances = [1000.,100.,10.,1.,0.1,0.01,0.001,0.0001]

wr(CSV)

graphs = load_graphs(folder=FOLDER_GRAPH , filename='graph-500-mu-0.1', n=50)

g = graphs[0]

for key in ['tolerance', 'graph'] + ['NMI', 'ARI','phi','gamma','Q','F1','F2','FM','JI','V','|c|','|g|','time']:
    wr(CSV, key)
wr(CSV,"\n")


for tol in tolerances:
    print("TOLERANCE", tol)
    for g_index, g in enumerate(graphs):

        #print(".", end="", flush=True)
        res = mpw_single_100(g, tol)
        #print("\n")

        wr(CSV, tol)
        wr(CSV,g_index )
        for key in ['NMI', 'ARI','phi','gamma','Q','F1','F2','FM','JI','V','|c|','|g|','time']:
            wr(CSV,res[key])
        wr(CSV, "\n")
#mpw_seq_lite(graphs, 0.001, repeats=10)


#!/usr/bin/python3

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def polarization_metric(model, country='USA', method='euclidean', savepath=None):

    M = model.vector_size
    dv = model.docvecs.offset2doctag
    if country=='USA':
        parliaments = [i for i in range(43,115)]
        years = [i for i in range(1873,2017,2)]
        party1 = ['D_' + str(p) for p in parliaments]
        party2 = ['R_' + str(p) for p in parliaments]
        label_to_year = {c:y for c,y in zip(parliaments, years)}
        assert len(party1)==len(party2)
    elif country=='UK':
        label_to_year = {37: '1935', 38: '1945', 39: '1950', 40: '1951', 41: '1955', 42: '1959', 43: '1964', 44: '1966',
                       45: '1970', 46: '1974.1', 47: '1974.2', 48: '1979', 49: '1983', 50: '1987', 51: '1992', 52: '1997',
                       53: '2001', 54: '2005', 55: '2010'}
        parliaments = [i for i in range(37,56)]
        years = [label_to_year[p] for p in parliaments]
        party1 = ['Lab_' + str(p) for p in parliaments]
        party2 = ['Con_' + str(p) for p in parliaments]
        assert len(party1)==len(party2)
    elif country=='Canada':
        label_to_year = {9.0: '1901', 10.0: '1904', 11.0: '1908', 12.0: '1911', 13.0: '1918', 14.0: '1922', 15.1: '1925.1', 15.2: '1925.2',
                        16.0: '1926', 17.0: '1930', 18.0: '1935', 19.0: '1940', 20.0: '1945', 21.0: '1949', 22.0: '1953', 23.0: '1957',
                        24.0: '1958', 25.0: '1962', 26.0: '1963', 27.0: '1965', 28.0: '1968', 29.0: '1972', 30.0: '1974', 31.0: '1979',
                        32.0: '1980', 33.0: '1984', 34.0: '1988', 35.0: '1993', 36.0: '1997', 37.0: '2000', 38.0: '2004', 39.0: '2006',
                        40.0: '2008', 41.0: '2011', 42.0: '2015'}
        parliaments = sorted(list(label_to_year.keys()))
        years = [label_to_year[p] for p in parliaments]
        party1 = ['Liberal_' + str(p) for p in parliaments]
        party2 = ['Conservative_' + str(p) for p in parliaments]
        assert len(party1)==len(party2)
    else:
        raise ValueError("Country must be 'USA', 'UK' or 'Canada'.")
    T = len(parliaments)
    parties = party1 + party2
    P = len(parties)
    z = np.zeros((P, M))
    for i in range(P):
        z[i,:] = model.docvecs[parties[i]]
    if method=='euclidean':
        D = euclidean_distances(z)[0:T,T:P].diagonal().tolist()
    elif method=='cosine':
        D = (1 - cosine_similarity)[0:T,T:P].diagoanl().tolist()
    else:
        raise ValueError("Method must be 'euclidean' or 'cosine'.")
    D = pd.DataFrame(D, columns=[method])
    D['parliament'] = parliaments
    D['year'] = years
    if savepath:
        D.to_csv(savepath, sep=',', index=False)
    return D[['parliament','year',method]]

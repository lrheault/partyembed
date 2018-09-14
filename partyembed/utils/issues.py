#!/usr/bin/python3

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def topic_vector(topicword, model, n = 20):

    M = model.vector_size
    sims = model.wv.most_similar(topicword, topn = n)
    simw = [topicword] + [w for w, s in sims]
    zsim = np.zeros((n + 1, M))
    for i, w in enumerate(simw):
        zsim[i,:] = model.wv[w]
    centroid = zsim.mean(axis=0)
    return centroid

def bootstrap_topic_vector(topicword, model, n = 20, sims=1000):

    M = model.vector_size
    expanded_word_list = model.wv.most_similar(topicword, topn = n-1)
    topic_words = [topicword] + [w for w, s in expanded_word_list]
    boot_results=np.zeros((sims,M))
    for s in range(sims):
        boot_sample = np.random.choice(topic_words, size=n)
        zsim = np.zeros((n, M))
        for i, w in enumerate(boot_sample):
            zsim[i,:] = model.wv[w]
        boot_results[s,:] = zsim.mean(axis=0)
    return boot_results

def cos_sim(parties, topic, boot=True, sims=1000):

    if boot:
        P = parties.shape[0]
        C = cosine_similarity(parties, topic)
        m = np.mean(C, axis=1)
        ci = np.percentile(C, q=[2.5, 97.5], axis=1)
        return m.reshape(P,).tolist(), ci[0].reshape(P,).tolist(), ci[1].reshape(P,).tolist()
    else:
        return cosine_similarity(parties, topic).reshape(P,).tolist()


def issue_ownership(model, topic_vector=None, topic_word=None, infer_vector=True, t_size = 20, boot=True, smooth=True, country='USA'):

    M = model.vector_size
    if topic_vector:
        t = topic_vector
    if topic_word:
        if infer_vector:
            if boot:
                t = bootstrap_topic_vector(topic_word, model, n = t_size, sims=1000)
            else:
                t = topic_vector(topic_word, model, n = t_size)
        else:
            t = model.wv[topic_word]
    res = fit(model, t, country=country, smooth=smooth, boot=boot)
    return res

def fit(model, topic_vector, country='USA', smooth=True, boot=True):

    M = model.vector_size
    dv = model.docvecs.offset2doctag
    if country=='USA':
        parliaments = [i for i in range(43,115)]
        years = [i for i in range(1873,2017,2)]
        parties = [d for d in dv if d.startswith('D_') or d.startswith('R_')]
        P = len(parties)
        z = np.zeros(( P, M ))
        for i in range(P):
            z[i,:] = model.docvecs[parties[i]]
        if boot:
            C, LB, UB = cos_sim(z, topic_vector, boot=True, sims=1000)
            res = pd.DataFrame({'congress': parliaments,
                                'year': years,
                                'dem': [s for t,s in zip(parties,C) if t.startswith('D_')],
                                'dem_lb': [s for t,s in zip(parties,LB) if t.startswith('D_')],
                                'dem_ub': [s for t,s in zip(parties,UB) if t.startswith('D_')],
                                'rep': [s for t,s in zip(parties,C) if t.startswith('R_')],
                                'rep_lb': [s for t,s in zip(parties,LB) if t.startswith('R_')],
                                'rep_ub': [s for t,s in zip(parties,UB) if t.startswith('R_')]})
        else:
            C = cos_sim(z, topic_vector, boot=False)
            res = pd.DataFrame({'congress': parliaments,
                                'year': years,
                                'dem': [s for t,s in zip(parties,C) if t.startswith('D_')],
                                'rep': [s for t,s in zip(parties,C) if t.startswith('R_')]})
        if smooth:
            res = res.rolling(window=5,center=False).mean()
            res['year'] = years
            res['congress'] = parliaments
        return res
    elif country=='UK':
        label_to_year = {37: '1935', 38: '1945', 39: '1950', 40: '1951', 41: '1955', 42: '1959', 43: '1964', 44: '1966',
                       45: '1970', 46: '1974.1', 47: '1974.2', 48: '1979', 49: '1983', 50: '1987', 51: '1992', 52: '1997',
                       53: '2001', 54: '2005', 55: '2010'}
        parliaments = [i for i in range(37,56)]
        years = [label_to_year[p] for p in parliaments]
        parties = [d for d in dv if 'Lab_' in d or 'Con_' in d or 'Lib_' in d]
        P = len(parties)
        z = np.zeros(( P, M ))
        for i in range(P):
            z[i,:] = model.docvecs[parties[i]]
        if boot:
            C, LB, UB = cos_sim(z, topic_vector, boot=True, sims=1000)
            res = pd.DataFrame({'parliament': parliaments,
                                'year': years,
                                'lab': [s for t,s in zip(parties,C) if 'Lab_' in t],
                                'lab_lb': [s for t,s in zip(parties,LB) if 'Lab_' in t],
                                'lab_ub': [s for t,s in zip(parties,UB) if 'Lab_' in t],
                                'con': [s for t,s in zip(parties,C) if 'Con_' in t],
                                'con_lb': [s for t,s in zip(parties,LB) if 'Con_' in t],
                                'con_ub': [s for t,s in zip(parties,UB) if 'Con_' in t],
                                'lib': [s for t,s in zip(parties,C) if 'Lib_' in t],
                                'lib_lb': [s for t,s in zip(parties,LB) if 'Lib_' in t],
                                'lib_ub': [s for t,s in zip(parties,UB) if 'Lib_' in t]})
        else:
            C = cos_sim(z, topic_vector, boot=False)
            res = pd.DataFrame({'parliament': parliaments,
                            'year': years,
                            'lab': [s for t,s in zip(parties,C) if 'Lab_' in t],
                            'con': [s for t,s in zip(parties,C) if 'Con_' in t],
                            'lib': [s for t,s in zip(parties,C) if 'Lib_' in t]})
        if smooth:
            res = res.rolling(window=5,center=False).mean()
            res['year'] = years
            res['parliament'] = parliaments
        return res
    elif country=='Canada':
        label_to_year = {9.0: '1901', 10.0: '1904', 11.0: '1908', 12.0: '1911', 13.0: '1918', 14.0: '1922', 15.1: '1925.1', 15.2: '1925.2',
                        16.0: '1926', 17.0: '1930', 18.0: '1935', 19.0: '1940', 20.0: '1945', 21.0: '1949', 22.0: '1953', 23.0: '1957',
                        24.0: '1958', 25.0: '1962', 26.0: '1963', 27.0: '1965', 28.0: '1968', 29.0: '1972', 30.0: '1974', 31.0: '1979',
                        32.0: '1980', 33.0: '1984', 34.0: '1988', 35.0: '1993', 36.0: '1997', 37.0: '2000', 38.0: '2004', 39.0: '2006',
                        40.0: '2008', 41.0: '2011', 42.0: '2015'}
        parliaments = sorted(list(label_to_year.keys()))
        years = [label_to_year[p] for p in parliaments]
        parties = [d for d in dv if 'Liberal_' in d or 'Conservative_' in d or 'NDP_' in d]
        P = len(parties)
        z = np.zeros(( P, M ))
        for i in range(P):
            z[i,:] = model.docvecs[parties[i]]
        if boot:
            C, LB, UB = cos_sim(z, topic_vector, boot=True, sims=1000)
            res = pd.DataFrame({'parliament': parliaments,
                                'year': years,
                                'lib': [s for t,s in zip(parties,C) if 'Liberal_' in t],
                                'lib_lb': [s for t,s in zip(parties,LB) if 'Liberal_' in t],
                                'lib_ub': [s for t,s in zip(parties,UB) if 'Liberal_' in t],
                                'con': [s for t,s in zip(parties,C) if 'Conservative_' in t],
                                'con_lb': [s for t,s in zip(parties,LB) if 'Conservative_' in t],
                                'con_ub': [s for t,s in zip(parties,UB) if 'Conservative_' in t],
                                'ndp': [np.nan]*10 + [s for t,s in zip(parties,C) if 'NDP_' in t],
                                'ndp_lb': [np.nan]*10 + [s for t,s in zip(parties,LB) if 'NDP_' in t],
                                'ndp_ub': [np.nan]*10 + [s for t,s in zip(parties,UB) if 'NDP_' in t]})
        else:
            C = cos_sim(z, topic_vector, boot=False)
            res = pd.DataFrame({'parliament': parliaments,
                            'year': years,
                            'lab': [s for t,s in zip(parties,C) if 'Liberal_' in t],
                            'con': [s for t,s in zip(parties,C) if 'Conservative_' in t],
                            'ndp': [np.nan]*10 + [s for t,s in zip(parties,C) if 'NDP_' in t]})
        if smooth:
            res = res.rolling(window=5,center=False).mean()
            res['year'] = years
            res['parliament'] = parliaments
        return res
    else:
        raise ValueError("Country must be 'USA', 'UK' or 'Canada'.")

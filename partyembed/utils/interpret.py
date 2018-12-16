#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from partyembed.utils.guided import BASE_LEXICON

class Interpret(object):
    
    def __init__(self, model, parties, dr, Z, labels, rev1=False, rev2=False, min_count=100, max_count = 1000000, max_features=10000):

        self.model = model
        self.parties = parties
        self.labels = labels
        self.P = len(self.parties)
        self.M = self.model.vector_size   
        self.voc = self.sorted_vocab(min_count, max_count, max_features)
        self.V = len(self.voc)   
        self.pca = dr
        self.max = Z.max(axis=0)
        self.min = Z.min(axis=0)
        self.sims = self.compute_sims()
        self.dim1 = rev1
        self.dim2 = rev2
        
    def sorted_vocab(self, min_count=100, max_count=10000, max_features=10000):
        wordlist=[]
        for word, vocab_obj in self.model.wv.vocab.items():
            wordlist.append((word, vocab_obj.count))
        wordlist = sorted(wordlist, key=lambda tup: tup[1], reverse=True)
        return [w for w,c in wordlist if c>min_count and c<max_count and w.count('_')<3][0:max_features]
    
    def compute_sims(self):

        Z = np.zeros((self.V, 2))
        for idx, w in enumerate(self.voc):
            Z[idx, :] = self.pca.transform(self.model.wv[w].reshape(1,-1))
        sims_right = euclidean_distances(Z, np.array([self.max[0],0]).reshape(1, -1))
        sims_left = euclidean_distances(Z, np.array([self.min[0],0]).reshape(1, -1))
        sims_up = euclidean_distances(Z, np.array([0,self.max[1]]).reshape(1, -1))
        sims_down = euclidean_distances(Z, np.array([0,self.min[1]]).reshape(1, -1))
        temp = pd.DataFrame({'word': self.voc, 'right': sims_right[:,0], 'left': sims_left[:,0], 'up': sims_up[:,0], 'down': sims_down[:,0]})
        return temp

    def combine(self):

        self.sims = self.sims.merge(self.dimreduce, on='party_label', how='left')
        dimscores = []
        for v in self.voc:
            c1=self.sims.dim1.corr(self.sims[v])
            c2=self.sims.dim2.corr(self.sims[v])
            dimscores.append((v,c1,c2))
        return pd.DataFrame(dimscores, columns=['word','corr1','corr2'])
        
    def top_words_list(self, topn=20):

        if self.dim1:
            temp = self.sims.sort_values(by='left')
            print("Words associated with positive values on dimension 1:\n")
            self.top_positive_dim1 = temp.word.tolist()[0:topn]
            self.top_positive_dim1 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim1])
            print(self.top_positive_dim1)
            print()
            temp = self.sims.sort_values(by='right')
            print("Words associated with negative values on dimension 1:\n")
            self.top_negative_dim1 = temp.word.tolist()[0:topn]
            self.top_negative_dim1 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim1])
            print(self.top_negative_dim1)
            print()
        else:
            temp = self.sims.sort_values(by='right')
            print("Words associated with positive values on dimension 1:\n")
            self.top_positive_dim1 = temp.word.tolist()[0:topn]
            self.top_positive_dim1 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim1])
            print(self.top_positive_dim1)
            print()
            temp = self.sims.sort_values(by='left')
            print("Words associated with negative values on dimension 1:\n")
            self.top_negative_dim1 = temp.word.tolist()[0:topn]
            self.top_negative_dim1 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim1])
            print(self.top_negative_dim1)
            print()
        if self.dim2:
            temp = self.sims.sort_values(by='down')
            print("Words associated with positive values on dimension 2:\n")
            self.top_positive_dim2 = temp.word.tolist()[0:topn]
            self.top_positive_dim2 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim2])
            print(self.top_positive_dim2)
            print()
            temp = self.sims.sort_values(by='up')
            print("Words associated with negative values on dimension 2:\n")
            self.top_negative_dim2 = temp.word.tolist()[0:topn]
            self.top_negative_dim2 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim2])
            print(self.top_negative_dim2)
        else:
            temp = self.sims.sort_values(by='up')
            print("Words associated with positive values on dimension 2:\n")
            self.top_positive_dim2 = temp.word.tolist()[0:topn]
            self.top_positive_dim2 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim2])
            print(self.top_positive_dim2)
            print()
            temp = self.sims.sort_values(by='down')
            print("Words associated with negative values on dimension 2:\n")
            self.top_negative_dim2 = temp.word.tolist()[0:topn]
            self.top_negative_dim2 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim2])
            print(self.top_negative_dim2)

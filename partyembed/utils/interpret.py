#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from partyembed.utils.guided import BASE_LEXICON

class Interpret(object):
    
    def __init__(self, model, parties, Z, labels, voc_size=20000, custom_vocab=None):

        self.model = model
        self.parties = parties
        self.labels = labels
        self.P = len(self.parties)
        self.M = self.model.vector_size   
        if custom_vocab:
            if custom_vocab=='default':
                self.V = len(BASE_LEXICON)
                selv.voc=[]
                for w in BASE_LEXICON:
                    if self.model.wv.vocab.get(w, None):
                        self.voc.append(w)
            elif type(custom_vocab)==list:
                self.V = len(custom_vocab)
                self.voc=[]
                for w in custom_vocab:
                    if self.model.wv.vocab.get(w, None):
                        self.voc.append(w)
            else:
                raise ValueError("Option custom_vocab must be either a list of words/phrases or 'default'.")
        else:
            self.V = voc_size
            self.voc = self.sorted_vocab()        
        self.sims = self.compute_sims()
        self.dimreduce = Z
        self.word_correlations = self.combine()
        
    def sorted_vocab(self):

        wordlist=[]
        for word, vocab_obj in self.model.wv.vocab.items():
            wordlist.append((word, vocab_obj.count))
        wordlist = sorted(wordlist, key=lambda tup: tup[1], reverse=True)
        return [w for w,c in wordlist[0:self.V]]
    
    def compute_sims(self):

        total_dim = self.P + self.V
        z = np.zeros((total_dim, self.M))     
        index = 0
        for p in self.parties:
            z[index,:] = self.model.docvecs[p]
            index+=1
        for v in self.voc:
            z[index,:] = self.model.wv[v]
            index+=1
        C = pd.DataFrame(cosine_similarity(z)[0:self.P,self.P:total_dim], columns = self.voc)
        C['party_label'] = self.labels
        return C

    def combine(self):

        self.sims = self.sims.merge(self.dimreduce, on='party_label', how='left')
        dimscores = []
        for v in self.voc:
            c1=self.sims.dim1.corr(self.sims[v])
            c2=self.sims.dim2.corr(self.sims[v])
            dimscores.append((v,c1,c2))
        return pd.DataFrame(dimscores, columns=['word','corr1','corr2'])
        
    def top_words(self, topn=20):

        temp = self.word_correlations.sort_values(by='corr1')
        print("Words positively correlated with dimension 1:\n")
        self.top_positive_dim1 = reversed(temp.word.tolist()[-topn:])
        for w in self.top_positive_dim1:
            w=w.replace('_',' ')
            print(w)
        print()
        print("Words negatively correlated with dimension 1:\n")
        self.top_negative_dim1 = temp.word.tolist()[0:topn]
        for w in self.top_negative_dim1:
            w=w.replace('_',' ')
            print(w)
        temp = self.word_correlations.sort_values(by='corr2')
        print()
        print("Words positively correlated with dimension 2:\n")
        self.top_positive_dim2 = reversed(temp.word.tolist()[-topn:])
        for w in self.top_positive_dim2:
            w=w.replace('_',' ')
            print(w)
        print()
        print("Words negatively correlated with dimension 2:\n")
        self.top_negative_dim2 = temp.word.tolist()[0:topn]
        for w in self.top_negative_dim2:
            w=w.replace('_',' ')
            print(w)

    def top_words_list(self, topn=20):

        temp = self.word_correlations.sort_values(by='corr1')
        print("Words positively correlated with dimension 1:\n")
        self.top_positive_dim1 = reversed(temp.word.tolist()[-topn:])
        self.top_positive_dim1 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim1])
        print(self.top_positive_dim1)
        print()
        print("Words negatively correlated with dimension 1:\n")
        self.top_negative_dim1 = temp.word.tolist()[0:topn]
        self.top_negative_dim1 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim1])
        print(self.top_negative_dim1)
        temp = self.word_correlations.sort_values(by='corr2')
        print()
        print("Words positively correlated with dimension 2:\n")
        self.top_positive_dim2 = reversed(temp.word.tolist()[-topn:])
        self.top_positive_dim2 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim2])
        print(self.top_positive_dim2)
        print()
        print("Words negatively correlated with dimension 2:\n")
        self.top_negative_dim2 = temp.word.tolist()[0:topn]
        self.top_negative_dim2 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim2])
        print(self.top_negative_dim2)

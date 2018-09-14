#!/usr/bin/python3

import pkg_resources
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import metrics
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
from partyembed.utils.labels import party_labels, party_tags
from partyembed.utils.guided import custom_projection_1D

DATA_PATH = pkg_resources.resource_filename('partyembed', 'data/')
MODEL_PATH = pkg_resources.resource_filename('partyembed', 'models/')

class Validate(object):

    def __init__(self, model, country='USA', method='pca', custom_lexicon=None):

        self.model = model
        self.M = self.model.vector_size
        self.country = country
        self.method = method
        self.custom_lexicon = custom_lexicon
        self.label_dict = party_labels(self.country)
        self.parties, _ = party_tags(self.model, self.country)
        self.labels = [self.label_dict[p] for p in self.parties]
        self.P = len(self.parties)
        self.components = 1
        self.placement = self.dimension_reduction()
        self.correlation, self.spearman = self.correlation_scores()
        self.p_accuracy = self.pairwise_accuracy()

    def dimension_reduction(self):

        z=np.zeros(( self.P, self.M ))
        for i in range( self.P ):
            z[i,:] = self.model.docvecs[self.parties[i]]
        if self.method=='pca':
            dr = PCA(n_components=self.components)
            Z = dr.fit_transform(z)
        elif self.method=='guided':
            Z = custom_projection_1D(z, self.model, custom_lexicon=self.custom_lexicon)
        else:
            raise ValueError("Method must be either pca or guided.")
        Z = pd.DataFrame(Z)
        Z.columns = ['score']
        Z['label'] = self.labels

        # Re-orienting the scale for substantive interpretation:
        if self.country=='USA':
            if Z[Z.label=='Dem 2015'].score.values[0] > Z[Z.label=='Rep 2015'].score.values[0]:
                Z['score'] = Z.score * (-1)
        elif self.country=='Canada':
            if Z[Z.label=='NDP 2015'].score.values[0] > Z[Z.label=='Cons 2015'].score.values[0]:
                Z['score'] = Z.score * (-1)
        else:
            if Z[Z.label=='Labour 2010'].score.values[0] > Z[Z.label=='Cons 2010'].score.values[0]:
                Z['score'] = Z.score * (-1)

        input_file = DATA_PATH + 'goldstandard_' + self.country.lower() + '.csv'
        ref = pd.read_table(input_file, sep=',',encoding='utf-8',header=0)
        ref = ref.merge(Z, on='label', how='left')

        return ref

    def accuracy(self, goldscores, testscores):

        gold=[]; test=[];
        for i in range(1,len(goldscores)):
            for j in range(i+1,len(goldscores)):
                if goldscores[i]>=goldscores[j]:
                    gold.append(1)
                else:
                    gold.append(0)
        for i in range(1,len(testscores)):
            for j in range(i+1,len(testscores)):
                if testscores[i]>=testscores[j]:
                    test.append(1)
                else:
                    test.append(0)
        return metrics.accuracy_score(gold, test)*100

    def correlation_scores(self):

        if self.country=='USA':
            return ([('voteview', self.placement.voteview.corr(self.placement.score))],
                    [('voteview', self.placement.voteview.corr(self.placement.score, method='spearman'))])
        else:
            return ([('rile', self.placement.rile.corr(self.placement.score)),
                    ('vanilla', self.placement.vanilla.corr(self.placement.score)),
                    ('legacy', self.placement.legacy.corr(self.placement.score))],
                    [('rile', self.placement.rile.corr(self.placement.score, method='spearman')),
                    ('vanilla', self.placement.vanilla.corr(self.placement.score, method='spearman')),
                    ('legacy', self.placement.legacy.corr(self.placement.score, method='spearman'))])

    def pairwise_accuracy(self):

        if self.country=='USA':
            acc = self.accuracy(self.placement.voteview.tolist(), self.placement.score.tolist())
            return [('voteview', acc)]
        else:
            acc1 = self.accuracy(self.placement.rile.tolist(), self.placement.score.tolist())
            acc2 = self.accuracy(self.placement.vanilla.tolist(), self.placement.score.tolist())
            acc3 = self.accuracy(self.placement.legacy.tolist(), self.placement.score.tolist())
            return [('rile', acc1),
                    ('vanilla', acc2),
                    ('legacy', acc3)]

    def benchmarks(self, test='analogies'):

        import logging
        logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
        if test=='analogies':
            self.model.wv.accuracy(DATA_PATH + 'questions-words.txt', restrict_vocab=10000)
        else:
            self.model.wv.evaluate_word_pairs(DATA_PATH + 'wordsim353.tsv')
        logging.basicConfig(level=logging.CRITICAL)

    def print_accuracy(self):

        print("Pearson Correlation Coefficient:")
        for d, c in self.correlation:
            print("%s: %0.3f" %(d,c))
        print()
        print("Spearman Rank Correlation Coefficient:")
        for d, c in self.spearman:
            print("%s: %0.3f" %(d,c))
        print()
        print("Pairwise Accuracy:")
        for d, c in self.p_accuracy:
            print("%s: %0.2f%%" %(d,c))
        print()

#!/usr/bin/python3

import pkg_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
from partyembed.utils.labels import party_labels, party_tags
from partyembed.utils.guided import custom_projection_2D
from partyembed.utils.polarization import polarization_metric
from partyembed.utils.interpret import Interpret
from partyembed.utils.issues import issue_ownership
from partyembed.validate import Validate

MODEL_PATH = pkg_resources.resource_filename('partyembed', 'models/')

class Explore(object):

    def __init__(self, model='House', method='pca', dimensions=2, country='USA', custom_lexicon=None):

        if type(model)==str:
            if model=='House':
                self.model = Doc2Vec.load(MODEL_PATH + 'house200')
                self.country = 'USA'
            elif model=='Senate':
                self.model = Doc2Vec.load(MODEL_PATH + 'senate200')
                self.country = 'USA'
            elif model=='USA':
                self.model = Doc2Vec.load(MODEL_PATH + 'usa200')
                self.country = 'USA'
            elif model=='Canada':
                self.model = Doc2Vec.load(MODEL_PATH + 'canada200')
                self.country = 'Canada'
            elif model=='UK':
                self.model = Doc2Vec.load(MODEL_PATH + 'uk200')
                self.country = 'UK'
            else:
                raise ValueError("Model must be House, Senate, Canada or UK, but you entered %s." % model)
        elif type(model)==Doc2Vec:
            self.model = model
            self.country = country
        else:
            raise ValueError("Model must be either a string or a Doc2Vec object.")
        self.custom_lexicon = custom_lexicon
        self.M = self.model.vector_size
        self.reverse = False
        self.method = method
        self.label_dict = party_labels(self.country)
        self.parties, self.cols = party_tags(self.model, self.country)
        self.labels = [self.label_dict[p] for p in self.parties]
        self.P = len(self.parties)
        self.components = dimensions
        self.placement = self.dimension_reduction()


    def dimension_reduction(self):

        z=np.zeros(( self.P, self.M ))
        for i in range( self.P ):
            z[i,:] = self.model.docvecs[self.parties[i]]
        if self.method=='pca':
            dr = PCA(n_components=self.components)
            Z = dr.fit_transform(z)
        elif self.method=='guided':
            Z = custom_projection_2D(z, self.model, custom_lexicon = self.custom_lexicon)
        else:
            raise ValueError("Model must be pca or guided.")
        Z = pd.DataFrame(Z)
        Z.columns = ['dim1', 'dim2']
        Z['party_label'] = self.labels

        # Re-orienting the scale for substantive interpretation:
        if self.country=='USA' and self.method!='guided':
            if Z[Z.party_label=='Dem 2015'].dim1.values[0] > Z[Z.party_label=='Rep 2015'].dim1.values[0]:
                Z['dim1'] = Z.dim1 * (-1)
                self.reverse = True
            if Z[Z.party_label=='Dem 2015'].dim2.values[0] < Z[Z.party_label=='Rep 2015'].dim2.values[0]:
                Z['dim2'] = Z.dim2 * (-1)
        if self.country=='Canada' and self.method!='guided':
            if Z[Z.party_label=='NDP 2015'].dim1.values[0] > Z[Z.party_label=='Cons 2015'].dim1.values[0]:
                Z['dim1'] = Z.dim1 * (-1)
                self.reverse = True
        if self.country=='UK' and self.method!='guided':
            if Z[Z.party_label=='Labour 2010'].dim1.values[0] > Z[Z.party_label=='Cons 2010'].dim1.values[0]:
                Z['dim1'] = Z.dim1 * (-1)
                self.reverse = True
        return Z

    def plot(self, axisnames=None, savepath=None):

        font = {'family' : 'Linux Libertine O',
        'weight' : 'normal',
        'size'   : 14}
        plt.rc('font', **font)
        plt.rc('axes', titlesize=20)
        plt.rc('axes', labelsize=20)
        plt.figure(figsize=(22,15))
        plt.scatter(self.placement.dim1, self.placement.dim2, color=self.cols)
        texts=[]
        for label, x, y, c in zip(self.labels, self.placement.dim1, self.placement.dim2, self.cols):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.3),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        if axisnames:
            plt.xlabel(axisnames[0])
            plt.ylabel(axisnames[1])
        else:
            if self.method=='guided':
                plt.xlabel("Economic Left-Right")
                plt.ylabel("Social Left-Right")
            else:
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
        if savepath:
            plt.savefig(savepath, dpi=600, bbox_inches='tight')
        plt.show()

    def interpret(self, top_words=20, voc_size=20000):
        Interpret(self.model, self.parties, self.placement, self.labels, voc_size=voc_size).top_words_list(top_words)

    def polarization(self):
        return polarization_metric(self.model, self.country)

    def issue(self, topic_word, lex_size=50):
        return issue_ownership(self.model, topic_word=topic_word, infer_vector=True, t_size=lex_size, country=self.country)

    def validate(self, custom_lexicon=None):
        Validate(self.model, self.country, self.method, custom_lexicon=custom_lexicon).print_accuracy()

    def benchmarks(self, test='analogies'):
        Validate(self.model, self.country, self.method).benchmarks(test=test)

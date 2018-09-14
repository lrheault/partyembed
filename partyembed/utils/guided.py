#!/usr/bin/python3

from gensim.models.doc2vec import Doc2Vec
from partyembed.utils.labels import party_labels, party_tags
import numpy as np
import pandas as pd
from sklearn import metrics

BASE_LEXICON = [['affordable_housing','decent_housing','eradicate_poverty','poverty','gap_rich_poor','wealthiest','low_income','inequality',
                'unequal','workers','minimum_wage','unemployment','unemployed','protective_tariff','redistribution','redistribution_wealth',
                'safety_net','social_security','homelessness','labor_unions','labour_unions','trade_unions','working_classes'],
                ['decentralization','bureaucracy','business','businesses','creating_jobs','job_creators','free_enterprise','free_trade',
                'private_enterprise','private_sector','debt_relief','debt_reduction','taxpayers','taxpayers_money','taxpayer_money',
                'commerce','privatisation','privatization','competitive','industry','productivity','deficit_reduction','hard_working','hardworking',
                'home_owners','homeowners','open_market','free_market','private_enterprise','private_sector','property_rights','property_owners'],
                ['minority_rights','gay_lesbian','affirmative_action','employment_equity','pay_equity','racial_minorities','racism','gun_control',
                'minorities','prochoice','pro-choice','civil_rights','environment','greenhouse_gas','pollution','climate_change','child_care',
                'childcare','planned_parenthood', 'access_abortion'],
                ['law_enforcement','moral_fabric','social_fabric','moral_decay','moral_values','sentences','tougher_sentences','traditional_values',
                'tradition','secure_borders','illegal_immigrants','illegal_immigration','criminals','fight_crime','prolife','pro-life',
                'sanctity_life','unborn_child','abortionist','church']]

def linear_projection_1D(pVec, vecXLeft, vecXRight):    
    vecX = vecXRight.mean(axis=0) - vecXLeft.mean(axis=0) 
    return np.dot(pVec, vecX)  

def linear_projection_2D(pVec, vecXLeft, vecXRight, vecYDown, vecYUp):    
    vecX = vecXRight.mean(axis=0) - vecXLeft.mean(axis=0) 
    vecY = vecYUp.mean(axis=0) - vecYDown.mean(axis=0)
    return (np.dot(pVec, vecX), np.dot(pVec, vecY)) 

def get_vector(model, words, M):
    words = [w for w in words if w in model.wv.vocab]
    L = len(words)
    temp = np.zeros((L, M))
    for i, x in enumerate(words):
        temp[i,:] = model.wv[x]
    return temp

def custom_projection_1D(z, model, custom_lexicon=None):
    M = model.vector_size
    if custom_lexicon:
        lex = custom_lexicon
        if len(lex)!=2:
            raise ValueError("The custom lexicon should be a list of lists, with two elements.")
    else:
        lex = [BASE_LEXICON[0] + BASE_LEXICON[2], BASE_LEXICON[1] + BASE_LEXICON[3]] 
    xl, xr = [get_vector(model, words, M) for words in lex] 
    projections = [linear_projection_1D(x, xl, xr) for x in z]
    Z = np.array(projections) 
    return Z

def custom_projection_2D(z, model, custom_lexicon=None):
    M = model.vector_size
    if custom_lexicon:
        lex = custom_lexicon
        if len(lex)!=4:
            raise ValueError("The custom lexicon should be a list of lists, with four elements.")
    else:
        lex = BASE_LEXICON
    xl, xr, yd, yu = [get_vector(model, words, M) for words in lex] 
    projections = [linear_projection_2D(x, xl, xr, yd, yu) for x in z]
    Z = np.array(projections) 
    return Z

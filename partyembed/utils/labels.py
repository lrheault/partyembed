#!/usr/bin/python3

from gensim.models.doc2vec import Doc2Vec

# Official party colors.
USA_COL = {'dem': '#3333FF', 'rep': '#E91D0E'}
CA_COL = {'lib': '#D71920', 'con': '#1A4782', 'ndp': '#F28000', 'bloc': '#33B2CC', 'ref': '#3CB371'}
UK_COL = {'lab': '#D50000', 'con': '#0087DC', 'lib': '#FDBB3A'}

# Full party names.
USA_NAMES = {'dem':'Democrats', 'rep':'Republicans'}
UK_NAMES = {'con':'Conservatives','lib':'Liberal-Democrats','lab':'Labour'}
CA_NAMES = {'bloc':'Bloc Quebecois', 'lib': 'Liberal','ref':'Reform-Alliance','con':'Conservatives','ndp':'NDP'}

def party_labels(country):

    if country=='USA':
        congress = [i for i in range(43,115)]
        years = [i for i in range(1873,2017,2)]
        usa_index_toyear = {c:y for c,y in zip(congress, years)}
        usa_labels = {'D_'+str(i): 'Dem '+str(usa_index_toyear[i]) for i in congress}
        usa_labels.update({'R_'+str(i): 'Rep '+str(usa_index_toyear[i]) for i in congress})
        return usa_labels
    elif country=='Canada':
        can_index_toyear = {9.0: '1901', 10.0: '1904', 11.0: '1908', 12.0: '1911', 13.0: '1918', 14.0: '1922', 15.1: '1925.1', 15.2: '1925.2',
        16.0: '1926', 17.0: '1930', 18.0: '1935', 19.0: '1940', 20.0: '1945', 21.0: '1949', 22.0: '1953', 23.0: '1957',
        24.0: '1958', 25.0: '1962', 26.0: '1963', 27.0: '1965', 28.0: '1968', 29.0: '1972', 30.0: '1974', 31.0: '1979',
        32.0: '1980', 33.0: '1984', 34.0: '1988', 35.0: '1993', 36.0: '1997', 37.0: '2000', 38.0: '2004', 39.0: '2006',
        40.0: '2008', 41.0: '2011', 42.0: '2015'}
        can_labels = {'Liberal_'+str(i): 'Liberal '+can_index_toyear[i] for i in can_index_toyear.keys()}
        can_labels.update({'Reform-Alliance_'+str(i): 'RefAll '+can_index_toyear[i] for i in can_index_toyear.keys()})
        can_labels.update({'Bloc_'+str(i): 'Bloc '+can_index_toyear[i] for i in can_index_toyear.keys()})
        can_labels.update({'Conservative_'+str(i): 'Cons '+can_index_toyear[i] for i in can_index_toyear.keys()})
        can_labels.update({'NDP_'+str(i): 'NDP '+can_index_toyear[i] for i in can_index_toyear.keys()})
        return can_labels
    elif country=='UK':
        uk_index_toyear = {37: '1935', 38: '1945', 39: '1950', 40: '1951', 41: '1955', 42: '1959', 43: '1964', 44: '1966',
        45: '1970', 46: '1974.1', 47: '1974.2', 48: '1979', 49: '1983', 50: '1987', 51: '1992', 52: '1997', 53: '2001', 54: '2005', 55: '2010'}
        uk_labels = {'Lab_'+str(i): 'Labour '+uk_index_toyear[i] for i in range(37,56)}
        uk_labels.update({'Lib_'+str(i): 'LibDems '+uk_index_toyear[i] for i in range(37,56)})
        uk_labels.update({'Con_'+str(i): 'Cons '+uk_index_toyear[i] for i in range(37,56)})
        return uk_labels
    else:
        raise ValueError("The country must be 'USA', 'Canada' or 'UK'.")

def party_tags(model, country):

    if country=='USA':
        democrats = [d for d in model.docvecs.offset2doctag if d.startswith('D_')] 
        republicans = [d for d in model.docvecs.offset2doctag if d.startswith('R_')] 
        parties = democrats + republicans
        cols = [USA_COL['dem']]*len(democrats) + [USA_COL['rep']]*len(republicans)
        fullnames = [USA_NAMES['dem']]*len(democrats) + [USA_NAMES['rep']]*len(republicans)
        return (fullnames, parties, cols)
    elif country=='Canada':
        ndp = [d for d in model.docvecs.offset2doctag if 'NDP' in d]
        bloc = [d for d in model.docvecs.offset2doctag if 'Bloc' in d]
        liberals = [d for d in model.docvecs.offset2doctag if 'Liberal' in d]
        conservatives = [d for d in model.docvecs.offset2doctag if 'Conservative' in d]
        reform = [d for d in model.docvecs.offset2doctag if 'Reform-Alliance' in d]
        parties = ndp + bloc + liberals + conservatives + reform
        cols = [CA_COL['ndp']]*len(ndp) + [CA_COL['bloc']]*len(bloc) + \
                [CA_COL['lib']]*len(liberals) + [CA_COL['con']]*len(conservatives) + [CA_COL['ref']]*len(reform)
        fullnames = [CA_NAMES['ndp']]*len(ndp) + [CA_NAMES['bloc']]*len(bloc) + \
                [CA_NAMES['lib']]*len(liberals) + [CA_NAMES['con']]*len(conservatives) + [CA_NAMES['ref']]*len(reform)
        return (fullnames, parties, cols)
    elif country=='UK':
        labour = [d for d in model.docvecs.offset2doctag if 'Lab' in d]
        liberals = [d for d in model.docvecs.offset2doctag if 'Lib' in d]
        conservatives = [d for d in model.docvecs.offset2doctag if 'Con' in d]
        parties = labour + liberals + conservatives 
        cols = [UK_COL['lab']]*len(labour) + [UK_COL['lib']]*len(liberals) + [UK_COL['con']]*len(conservatives)
        fullnames = [UK_NAMES['lab']]*len(labour) + [UK_NAMES['lib']]*len(liberals) + [UK_NAMES['con']]*len(conservatives)
        return (fullnames, parties, cols)
    else:
        raise ValueError("The country must be 'USA', 'Canada' or 'UK'.")

#!/bin/python
from random import randint
from gensim import utils, matutils
import gensim
from collections import defaultdict
import os.path
import argparse
from gensim.models import word2vec
import marshal
import multiprocessing
from multiprocessing.pool import ThreadPool
import itertools
from itertools import chain, groupby
from operator import itemgetter
import dawg
from numpy import exp, dot, zeros, outer, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, mean as npmean

import random
import sys
import csv
#from collections import Counter
from threading import RLock
import threading
from joblib import Parallel, delayed


def startswith(trie, prefix, exclude=[]):
    return [ w for w in trie.keys(prefix) if w not in exclude and len(w)-len(prefix) >= min_word_length ]

def endswith(trie, suffix, exclude=[]):
    exclude = map(lambda w: w[::-1], exclude)
    return [ w[::-1] for w in trie.keys(suffix[::-1]) if w not in exclude and len(w)-len(suffix) >= min_word_length ]

def word_without_prefix(w, length):
    return w[length:]

def word_without_suffix(w, length):
    if length == 0:
        return w
    else:
        return w[:-length]


max_affix_length=6
min_word_length=2

global_rule_map = {}

def rules_only(ws, voc_trie, voc_trie_rev, lookup):
  rules = defaultdict(int)
  for w1 in ws:
      for l in range(0, max_affix_length):
          if len(w1) - l < min_word_length:
              break
          w1_nosuffix = word_without_suffix(w1, l)
          for w2 in startswith(voc_trie, w1_nosuffix, exclude=[w1]): #Suffix
              if len(w2) - len(w1_nosuffix) < max_affix_length:
                  extracted_rule = rule_to_int("-"+suffix(w1, l), "-"+suffix(w2, len(w2) - len(w1_nosuffix)), lookup) #lookup["-"+suffix(w1, l)] * len(lookup) + lookup["-"+suffix(w2, len(w2) - len(w1_nosuffix))]
                  rules[extracted_rule] += 1
          w1_nopre = word_without_prefix(w1, l)
          for w2 in endswith(voc_trie_rev, w1_nopre, exclude=[w1]): #Prefix an- heuern, raus- heuern
              if len(w2) - len(w1_nopre) < max_affix_length:
                  extracted_rule = rule_to_int(prefix(w1, l)+"-", prefix(w2, len(w2) - len(w1_nopre))+"-", lookup) #lookup[prefix(w1, l)+"-"] * len(lookup) + lookup[prefix(w2, len(w2) - len(w1_nopre))+"-"]
                  rules[extracted_rule] += 1
  sys.stderr.write('.')
  return rules

def support(ws, voc_trie, voc_trie_rev, lookup, all_rules):
  support_set = defaultdict(list)

  def add_support(rule, w1, w2):
    if rule in all_rules:
      pairid = pair_to_int(w1, w2)
      if len(support_set[rule]) < SAMPLE_SIZE:
        support_set[rule].append(pairid)
      else:
        if randint(0, 10) < 1: #add some randomness
           i = randint(0, SAMPLE_SIZE-1)
           support_set[rule][i] = pairid

  for w1 in ws:
      for l in range(0, max_affix_length):
          if len(w1) - l < min_word_length:
              break
          w1_nosuffix = word_without_suffix(w1, l)
          for w2 in startswith(voc_trie, w1_nosuffix, exclude=[w1]): #Suffix
              if len(w2) - len(w1_nosuffix) < max_affix_length:
                  extracted_rule = lookup["-"+suffix(w1, l)] * len(lookup) + lookup["-"+suffix(w2, len(w2) - len(w1_nosuffix))]
                  add_support(extracted_rule, w1, w2)
          w1_nopre = word_without_prefix(w1, l)
          for w2 in endswith(voc_trie_rev, w1_nopre, exclude=[w1]): #Prefix an- heuern, raus- heuern
              if len(w2) - len(w1_nopre) < max_affix_length:
                  extracted_rule = lookup[prefix(w1, l)+"-"] * len(lookup) + lookup[prefix(w2, len(w2) - len(w1_nopre))+"-"]
                  add_support(extracted_rule, w1, w2)
  sys.stderr.write('.')
  return support_set

SAMPLE_SIZE=1000

def sample_pairs(pairs):
    return random.sample(pairs, min(SAMPLE_SIZE, len(pairs)))

def merge_rule_counts(dict_list):
    final_table = {}
    print "Merging dicts..."
    for i in range(0, len(dict_list)):
        dict = dict_list[i]
        for (k, v) in dict.iteritems():
            final_table[k] = final_table.get(k, 0) + v
        dict_list[i]=None
        sys.stderr.write('.')

    print "Cleaning the final dict..."
    delete = [k for (k, v) in final_table.iteritems() if v < 50]

    print "Keys to delete: ", len(delete)
    for k in delete:
        del final_table[k]

    return final_table

def merge_support(dict_list):
    final_table = {}
    print "Merging dicts..."
    for dict_i in range(0, len(dict_list)):
        dict = dict_list[dict_i]
        for (k, v) in dict.iteritems():
          if k not in final_table:
            final_table[k] = v
          else:
              for pairid in v:
                  if len(final_table[k]) < SAMPLE_SIZE:
                      final_table[k].append(pairid)
                  else:
                      if randint(0, 10) < 1: #add some randomness
                          i = randint(0, SAMPLE_SIZE-1)
                          final_table[k][i] = pairid
        dict_list[dict_i]=None
        sys.stderr.write('.')

    return final_table

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def word_in_topn(w1, w2, w3, w4, rule, n):
    #topn = model.most_similar(positive=[w1, w4], negative=[w3], topn=100)
    positive=[(w1, 1.0), (w4, 1.0)]
    negative=[(w3, -1.0)]

    model.init_sims()

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        mean.append(weight * model.syn0norm[model.vocab[word].index])
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    w2_similarity = float(dot(matutils.unitvec(model.syn0norm[model.vocab[w2].index]), mean))

    better = 0
    for sim in model.syn0norm:
        if float(dot(sim, mean)) > w2_similarity:
            better += 1
        if better > n:
            return False

    #better is still smaller than n, so it's in the best n words
    return True

def hit_rate((rule, support)):
   return npmean( map(lambda p: hit_rate_pair(rule, support, p)[0], support) )


def suffix(w, l):
    if l == 0:
        return ""
    return w[-l:]

def prefix(w, l):
    if l == 0:
        return ""
    return w[:l]

def int_to_rule(rule_id, lookup, lookup_rev):
    a2i = rule_id % len(lookup)
    a1i = (rule_id - a2i) / len(lookup)
    return (lookup_rev[a1i], lookup_rev[a2i])
 
def rule_to_int(affix1, affix2, lookup):
   return lookup[affix1] * len(lookup) + lookup[affix2]

def hit_rate_pair(rule, support, pair, idx_annoy):
   hits = 0
   totals = 0
   (w1, w2) = pair
   hit_pairs = set()
   for (w3, w4) in support:
       if (w1, w2) != (w3, w4):
            if word_in_topn_approx(w1, w2, w3, w4, rule, 100, idx_annoy):
                hit_pairs.add((w3, w4))
                hits += 1
            totals += 1
   hitrate = 0.0 if totals == 0 else hits / float(totals)
   return (hitrate, hit_pairs)



def extract_transforms(ruleid, support, idx_annoy):

    support_set = set([int_to_pair(pairid) for pairid in support])
    prototypes = []

    explains = dict([(pair, hit_rate_pair(ruleid, support_set, pair, idx_annoy)[1]) for pair in support_set])
    sys.stderr.write('h')

    while True:
        explains_by_count = sorted(explains.items(), key=lambda (k,v): -len(v))
        best = explains_by_count[0]
        if len(best[1]) >= 10: #The prototype explains more than 10 word pairs
            prototypes.append((best[0][0], len(best[1]) / float(len(support_set))))
        else:
            break
        del explains[best[0]]
        #Remove all explained pairs from the support set
        support_set = support_set - best[1]
        for k, v in explains.items():
            explains[k] = explains[k] - best[1]
        explains_by_count.pop(0)

        if not (len(support_set) >= 10 and len(explains_by_count) and explains_by_count[0] >= 10):
            break
    sys.stderr.write('Extracted prototypes for '+ str(ruleid) + ': \n')
    for pr in prototypes:
        sys.stderr.write(" - " + str(pr) + "\n")
    return (ruleid, prototypes)

def word_in_topn_approx(w1, w2, w3, w4, rule, n, idx_annoy):
    positive=[(w1, 1.0), (w4, 1.0)]
    negative=[(w3, -1.0)]
 
    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        mean.append(weight * model.syn0norm[model.vocab[word].index])
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    #w2_similarity = float(dot(matutils.unitvec(model.syn0norm[model.vocab[w2].index]), mean))
    idx_w2 = model.vocab[w2].index

    best_by_nn = idx_annoy.get_nns_by_vector(list(mean), 100)

    #better is still smaller than n, so it's in the best n words
    return idx_w2 in best_by_nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('embeddings', type=str)
    parser.add_argument('--run', type=str, default='run')
    
    args = parser.parse_args()
    
    #Load the model
    model = word2vec.Word2Vec.load(args.embeddings)
    
    #Vocabulary
    V = [ v for v in model.vocab.keys() if not v.startswith("DBPEDIA") ]#[:1000]
    V_lookup = dict(enumerate(V))
    V = map(unicode, V)
    
    print "Loaded vocab of size ", len(V)
    
    print "Building forward trie"
    
    #Build a trie
    voc_trie = dawg.CompletionDAWG(V)
    
    print "Building backward trie"
    voc_trie_rev = dawg.CompletionDAWG(map(lambda v: v[::-1], V))
    
    print "Done."
    
    if not os.path.exists(args.run):
      os.mkdir(args.run)
      print "Creating run " + args.run
 
    print "Creating Voc lookup"
    (voc_lookup, voc_lookup_rev) = ({}, {})
    if os.path.exists(args.run + '/voc_lookup.dat'):
        (voc_lookup, voc_lookup_rev) = marshal.load(open(args.run + '/voc_lookup.dat', 'rb'))
    else:
        for w in V:
            voc_lookup[w] = len(voc_lookup)
            voc_lookup_rev[voc_lookup[w]] = w
        print "Lookup size:", len(voc_lookup)
        marshal.dump((voc_lookup, voc_lookup_rev), open(args.run + '/voc_lookup.dat', 'wb'))
    print "Done."
    
    def pair_to_int(w1, w2):
        return voc_lookup[w1] * len(voc_lookup) + voc_lookup[w2]
    
    def int_to_pair(pairid):
        w2i = pairid % len(voc_lookup)
        w1i = (pairid - w2i) / len(voc_lookup)
        return (voc_lookup_rev[w1i], voc_lookup_rev[w2i])
    
    print "Extracting suffixes and prefixes"
    lookup = {}
    if os.path.exists(args.run + '/lookup.dat'):
        lookup = marshal.load(open(args.run + '/lookup.dat', 'rb'))
    else:
        for w in V:
            for l in range(0, max_affix_length):
                if len(w) - l < min_word_length:
                    break
                suf = "-"+suffix(w, l)
                pre = prefix(w, l)+"-"
                if suf not in lookup:
                    lookup[suf] = len(lookup)
                if pre not in lookup:
                    lookup[pre] = len(lookup)
        print "Lookup size:", len(lookup)
        marshal.dump(lookup, open(args.run + '/lookup.dat', 'wb'))
    print "Done."
    lookup_rev = dict([ (v,k) for (k,v) in lookup.iteritems() ])
    
    def extract_rules(ws):
        return rules_only(ws, voc_trie, voc_trie_rev, lookup)
    pool = multiprocessing.Pool(processes=22)
    
    print "Extracting candidates"
    #Extract candidate rules
    all_rules={}
    if os.path.exists(args.run + '/rules.dat'):
        all_rules = marshal.load(open(args.run + '/rules.dat', 'rb'))
    else:
        all_rules = merge_rule_counts(pool.map(extract_rules, list(chunks(V, 20000))))
        print "Extracted %d rules." % len(all_rules)
        marshal.dump(all_rules, open(args.run + '/rules.dat', 'wb'))
    
    print "Gathering support pairs"
    
    def extract_support(ws):
      return support(ws, voc_trie, voc_trie_rev, lookup, all_rules)
    pool_support = multiprocessing.Pool()
    
    if os.path.exists(args.run + '/support.dat'):
        all_rules_with_support = marshal.load(open(args.run + '/support.dat', 'rb'))
    else:
        all_rules_with_support = merge_support(pool_support.map(extract_support, list(chunks(V, 20000))))
        marshal.dump(all_rules_with_support, open(args.run + '/support.dat', 'wb'))

    print "Top 10 rules:"
    rs=sorted(all_rules.items(), key= lambda (k,v): -v)
    for (r, c) in rs[:10]:
        print int_to_rule(r, lookup, lookup_rev), c

    print "Bottom 10 rules:"
    for (r, c) in rs[-10:]:
        print int_to_rule(r, lookup, lookup_rev), c

    print "Rules with c>1000:", len(filter(lambda r: r[1] >= 1000, rs))
    print "Rules with c>500:",  len(filter(lambda r: r[1] >= 500, rs))
    print "Rules with c>100:",  len(filter(lambda r: r[1] >= 100, rs))

    restricted_rule_set = map(lambda r: r[0], filter(lambda r: r[1] >= 500, rs))
    restricted_rules_with_support = dict(filter(lambda p: p[0] in restricted_rule_set, all_rules_with_support.iteritems()))

    print "Extracting morphological transformations"
 
    
    
    #Normalized vectors need to be initialized
    model.init_sims()

    import annoy
    num_features = model.layer1_size
    idx_annoy = annoy.AnnoyIndex(num_features, metric='angular')
    idx_file = args.run + '/annoy.idx'
    if os.path.exists(idx_file):
        print "Loading similarity search index..."
        idx_annoy.load(idx_file)
    else:
        print "Building similarity search index..."
        for i, vec in enumerate(model.syn0norm):
            idx_annoy.add_item(i, list(vec))
        idx_annoy.build(500)
        idx_annoy.save(idx_file)
    print "Done."

    def extract_transforms_approx(p):
        return extract_transforms(p[0], p[1], idx_annoy)
   
    #print extract_transforms_approx(restricted_rules_with_support.items()[0])
    pool3 = multiprocessing.Pool()
    transforms = list(pool3.map(extract_transforms_approx, restricted_rules_with_support.items()))
    marshal.dump(transforms, open(args.run + '/transforms.dat', 'wb'))
    
    #print "Building graph"
    
    

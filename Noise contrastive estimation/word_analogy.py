import os
import pickle
import numpy as np
import gc
from sklearn.metrics.pairwise import cosine_similarity
import operator


model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

def get_diff(wordPair):
    tokens = wordPair.split(':')
    vec1 = embeddings[dictionary[tokens[0].replace('"', "").strip()]]
    vec2 = embeddings[dictionary[tokens[1].replace('"', "").strip()]]
    diff = vec1 - vec2
    return diff

def get_cosine(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]


def get_min_max(correct, test):
    hm = {}
    for tpair in test:
        hm[tpair] = 0
        for cpair in correct:
            hm[tpair] += get_cosine(test[tpair], correct[cpair])

    sorted_d = sorted(hm.items(), key=operator.itemgetter(1))
    return sorted_d[-1][0], sorted_d[0][0]


def process_data(file_name, suffix=''):
    # Read the file and produce the output
    fptrI = open('word_analogy_dev.txt', 'r')
    fptrO = open('word_analogy_dev_output_'+loss_model+suffix+'.txt', 'w')
    lines = fptrI.readlines()
    print ("Processing ...")
    n = len(lines)
    i = 0
    prev = 0
    for line in lines:
        percent = i *100// n
        i += 1
        if (percent % 10 == 0 and prev != percent):
            print (str(percent)+"% processed ")
        prev = percent
        tokens = line.split('||')
        true = tokens[0]
        toPred = tokens[1]
        # Write the first four to the output file
        printStr = tokens[1].replace(',', ' ').strip()
        toPred = tokens[1].split(',')
        true = tokens[0].split(',')
        # Get differences for the correct pairs
        correct = {}
        for pairs in true:
            correct[pairs] = get_diff(pairs)
        # Get difference for the test pairs
        test = {}
        for pairs in toPred:
            test[pairs] = get_diff(pairs)

        # Get max and min pairs
        max_pair, min_pair = get_min_max(correct, test)
        printStr = printStr + " " +max_pair.strip() + " " +min_pair.strip()+"\n"
        fptrO.write(printStr)
        gc.collect()
    fptrO.close()
    fptrI.close()


process_data('first_output', suffix='batch64')

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

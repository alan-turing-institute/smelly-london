# Script for extracting implied references to smells from  Medical Officer of Health (MOH) reports from the Greater London area spanning from 1848 to 1972,
# as part of the project "Smelly London" http://londonsmells.co.uk
# Author: Barbara McGillivray
# Version: 1.1
# Changes from version 1.0: investigated contexts of "effluent"
#https://radimrehurek.com/gensim/tut1.html
#http://www.katrinerk.com/courses/introduction_to_computational_linguistics_spring_2012/ics12_schedule/python-code-creating-a-vector-space-representation

# ------------------------
# Initialization:
# ------------------------

# Import libraries:
from __future__ import division
import nltk
import math
import os
from os import listdir
from os.path import isfile, join
import codecs
from nltk import word_tokenize
import csv
from nltk.corpus import stopwords
import re
from datetime import datetime

# Directory and file names:

#dir_in = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "selected"))  # relative path to data directory
dir_in = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Full text"))
dir_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "output"))  # relative path to data directory
file_out = "distances_smellwords_impliedwords_context100.csv"
file_out_context = "smell_synonyms_context.txt"

# create output directory if it doesn't exist:
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Read input files:

files = [f for f in listdir(dir_in) if isfile(join(dir_in, f)) and f.endswith('.txt') ]#and f.startswith('Acton')]
#files = files[0:20]

SMELL_WORDS = ['smell', 'stench', 'stink', 'odour', 'sniff', 'effluvium', 'aroma', 'pungent', 'pungency']
IMPLIED_SMELL_WORDS = ['effluent', 'bakehouse', 'trade', 'fumigate', 'sewer', 'furnace', 'disinfect', 'rat', 'nuisance',
                       'fly', 'rubbish', 'wind', 'vessel', 'offensive']


# Function for converting WordNet lemmatizer PoS tags into more standard PoS tags:

def get_new_pos(old_pos):

    new_pos = ""

    if old_pos.startswith('J'):
        new_pos = "a"
    elif old_pos.startswith('V'):
        new_pos = "v"
    elif old_pos.startswith('N'):
        new_pos = "n"
    elif old_pos.startswith('R'):
        new_pos = "r"
    else:
        new_pos = ""

    return new_pos


# Pre-process text:

wnl = nltk.WordNetLemmatizer()
words = []

for file in files:
    print("file:" + file)
    text = codecs.open(os.path.join(dir_in, file), 'r').read()
    tokens = word_tokenize(text)
    pos_tagging = nltk.pos_tag(tokens)
    for (word, pos) in pos_tagging:
        wordnet_pos = get_new_pos(pos)
        if wordnet_pos != "":
            lemma = wnl.lemmatize(word.lower(), wordnet_pos)
        else:
            lemma = word.lower()

        if word.istitle():
            lemma = lemma.capitalize()
        elif word.upper() == word:
            lemma = lemma.upper()
        #print("word:"+word)
        #print("pos:"+str(pos))
        #print("lemma:"+lemma)

    #lemmas = [wnl.lemmatize(w[0], get_new_pos(w[1])) for w in pos_tagging]
        words.append(lemma)
    #print(str(words))
    #words = words + tokens

#brown_words = list(nltk.corpus.brown.words())

print("computing space...")

stoplist = set(stopwords.words('english'))
context_size = 100
space = nltk.ConditionalFreqDist()

for index in range(len(words)):

    # current word
    current = words[ index ]

    # context before the current word: count each item
    # but no preceding context for index 0
    if index > 0:
        # don't start from a cxword_index < 0 in case index < context_size

        for cxword_index in range(max(index - context_size, 0), index):
            cxword = words[ cxword_index ]

            # In a ConditionalFreqDist, if 'current' is not a condition yet,
            # then accessing it creates a new empty FreqDist for 'current'
            # The FreqDist method inc() increments the count for the given item by one.
            # Change from Python 2.7 to Python 3.5: fdist.inc(x) → fdist[x] += 1
            # space[ current ].inc(cxword)

            # Exclude stopwords from context:
            if cxword not in stoplist and cxword not in [",",".",";",":","?","!", "(", ")"] and not re.match("^\d+$", cxword):
                space[current][cxword] += 1

    # context after the current word: count each item
    # but no succeeding context for the last item (index len(brown_words - 1))
    if index < len(words) - 1:

        # don't run until a cxword_index > len(brown_words) in case
        # index + context_size > len(brown_words)
        for cxword_index in range(index + 1, min(index + context_size + 1, len(words))):

            cxword = words[ cxword_index ]

            # In a ConditionalFreqDist, if 'current' is not a condition yet,
            # then accessing it creates a new empty FreqDist for 'current'
            # The FreqDist method inc() increments the count for the given item by one.
            # Change from Python 2.7 to Python 3.5: fdist.inc(x) → fdist[x] += 1
            #space[ current ].inc(cxword)

            # Exclude stopwords from context:
            if cxword.lower() not in stoplist and cxword not in [",",".",";",":","?","!", "(", ")"] and not re.match("^\d+$", cxword):
                space[current][cxword] += 1

#print("COUNTS FOR 'state':")
##for cxword, count in space[ 'election' ].items()[:50]:
#for cxword, count in list(space[ 'state' ].items())[:10]:
#    print(cxword, ":", count)

#print("COUNTS FOR 'cleanly':")
##for cxword, count in space[ 'water' ].items()[:50]:
#for cxword, count in list(space[ 'cleanly' ].items())[:10]:
#    print(cxword, ":", count)


# cosine similarity between word1 and word2:
#
# sum_w space[word1][w] * space[word2][w]
# -----------------------------------------------------------
# sqrt(sum_w space[word1][w]^2) * sqrt(sum_w space[word2][w]^2)
#
def cosine(space, word1, word2):
    denominator = math.sqrt(sum([count*count for count in space[word1].values()])) * math.sqrt(sum([count * count for count in space[word2].values()]))
    numerator = sum([ space[word1][w] * space[word2][w] for w in space[word1].keys() ])
    #return float(numerator) / float(denominator)
    cosine = 0
    error = 0
    try:
        cosine = float(numerator) / float(denominator)
    except:
        error = 1
        #print("Denominator is zero!")

    return cosine


# some word similarities

#print(cosine(space, "cleanly", "state"))
#print(cosine(space, "diagnosis", "walls"))

# Find the words most similar to "smell" words:

output = csv.writer(open(os.path.join(dir_out, file_out), 'w'), delimiter=',', quoting=csv.QUOTE_MINIMAL)
output.writerow(['target_word', 'synonym',  'cosine_distance'])

# output_context = open(os.path.join(dir_out, file_out_context), 'w')
# 
# # Calculate similarity between smell words and annotated implied smell words:
# 
# 
# start=datetime.now()
# 
# # Distance between smell words and implied smell words:
# 
# cosine_average = 0
# cosine_sum = 0
# count = 0
# for implied in IMPLIED_SMELL_WORDS:
#     print(implied)
#     for smell_word in SMELL_WORDS:
#         print(smell_word)
#         count += 1
#         cosine_implied_smell = cosine(space, implied, smell_word)
#         cosine_sum = cosine_sum + cosine_implied_smell
#         print(cosine_implied_smell)
#         output.writerow([implied, smell_word, cosine_implied_smell])
# 
# print(str(cosine_sum))
# print(str(count))
# cosine_average = round(cosine_sum/count,10)
# print(str(cosine_average))
# output.writerow([cosine_average])
# #output.close()
# print(datetime.now()-start)

# Distance between smell words and all words, with a threshold at 0.20:


# for smell_word in SMELL_WORDS:
#     smell_words_cosine_all = [(t,cosine(space,smell_word,t)) for t in set(words) if cosine(space, smell_word, t) != "" and cosine(space, smell_word, t) > 0]
#     smell_words_cosine = sorted(smell_words_cosine_all,key=lambda x:(-x[1],x[0]))
#     smell_words_cosine = [(w,c) for (w,c) in smell_words_cosine if c > 0.2 and w != smell_word]
#     smell_words = [w for (w,c) in smell_words_cosine]
# 
#     # only keep nouns:
# 
#     smell_words_pos = [nltk.pos_tag([w]) for w in smell_words]
# 
#     for i in range(len(smell_words_cosine)):
#         if get_new_pos(smell_words_pos[i][0][1]) in ["n"]:  # "["ADJ", "N", "V"]:
#             print(smell_words_cosine[i][0], ":", smell_words_cosine[i][1])
#             output.writerow([smell_word, smell_words_cosine[i][0], smell_words_cosine[i][1]])
#             #output_context.write("Contexts for " + smell_words_cosine[i][0] + "\n")
#             #for cxword, count in list(space[smell_words_cosine[i][0]].items()):
#             #    print(cxword, ":", count)
#             #output.writerow([smell_word, smell_words_cosine[i][0], smell_words_cosine[i][1]])
#             #    output_context.write(cxword + ":" + str(count) + "\n")
            
output.close()
#This script outputs the closest words by cosine to the mean vector of a set of input words in a pre-established word2vec model.

import gensim

fr = "/path/to/output/model

def returner(words,meth,win,dim,lim):
    mod = gensim.models.word2vec.Word2Vec.load("/home/masteradamo/academy/models/WellcomeMOH/moh2vec/" + meth + win + "x" + dim)
    return [x[0] for x in mod.most_similar(positive=words,topn=int(lim))]

import collections
import numpy as np
#from nltk.stem import SnowballStemmer

f = "/home/masteradamo/academy/models/WellcomeMOH/"
fr = "/home/masteradamo/academy/models/WellcomeMOH/2x2/wordus.txt"

#stemmer = SnowballStemmer("english",ignore_stopwords=True)

def translater():
    rt = [x.split("::")[0] for x in open(fr,'r').readlines()]
    tr = {rt[n]:str(n) for n in range(len(rt))}
    return tr,rt

def indyer(inds,ext):
    outs = set()
    pt = 0
    while len(outs) < ext:
#        print "LOUT",outs,pt,len(outs)
        pt  += max(1,(ext-len(outs))/len(inds))
        outs.update([x for y in inds for x in y[:pt]])
    return outs

def spacer(words,win,ext):
    wecs = []
    vecs = collections.defaultdict(list)
    inds = []
    fv = f + win + "x" + win + "/vectors/"
    for word in words:
        indl = []
        vec = collections.defaultdict(float)
        for pair in open(fv+word+".txt",'r').readlines():
            vecs[pair.split("::")[0]].append(float(pair.split("::")[1]))
            indl.append((float(pair.split("::")[1]),pair.split("::")[0]))
            vec[pair.split("::")[0]] = float(pair.split("::")[1])
        inds.append([x[1] for x in sorted(indl,reverse=True)[:ext]])
        wecs.append(vec)
    joint = [x[1] for x in sorted([(sum(vecs[y]),y) for y in vecs if len(vecs[y])==len(words)],reverse=True)][:ext]
    simple = [x[1] for x in sorted([(sum(vecs[y]),y) for y in vecs],reverse=True)][:ext]
    indy = indyer(inds,ext)
    return [joint,indy],wecs

def measurer(space,vecs,words,win):
    mean = {x:(sum([y[x] for y in vecs])/len(words)) for x in space}
    morm = np.sqrt(sum([mean[x]**2 for x in mean]))
    vorm = np.mean([np.sqrt(sum([y[x]**2 for x in space])) for y in vecs])
    rat = vorm/morm
    print "RAT",rat
    mean = {x:mean[x]*rat for x in mean}
    norms = collections.defaultdict(float)
    dists = {}
    seen = 0.0
    fd = f + win + "x" + win + "/dimensions/"
    for dim in space:
#        print "DIM",dim
        pairs = open(fd+dim+".txt",'r').readlines()
#        print "PAIRS",pairs
        hits = {}
        for pt in pairs:
#            print "PT",pt
            vec = pt.split("::")[0]
            val = float(pt.split("::")[1])
            if pt.split("::")[0] not in words:
                norms[vec] += val**2
                if vec not in dists:
                    dists[vec] = seen
                dists[vec] += (mean[dim]-val)**2
                hits[vec] = True
        for item in dists:
            if item not in hits:
                dists[item] += mean[dim]**2
        seen += mean[dim]**2
    norm = [x[1] for x in sorted([(norms[y],y) for y in norms],reverse=True)]
    dist = [x[1] for x in sorted([(dists[y],y) for y in dists])]
    return norm,dist

def taker(words,win,ext,lim):
    words = [tr[x] for x in words]
    spaces,vecs = spacer(words,win,int(ext))
    print "SPACES PROJECTED"
    names = ["JOINT","INDY"]
    turns = []
    for n in range(len(spaces)):
        norm,dist = measurer(spaces[n],vecs,words,win)
        print names[n]
        print "BY NORM",[rt[int(x)] for x in norm[:int(lim)]]
        print "BY DISTANCE",[rt[int(x)] for x in dist[:int(lim)]]
        turns.append([[rt[int(x)] for x in norm[:int(lim)]],[rt[int(x)] for x in dist[:int(lim)]]])
    return turns

tr,rt = translater()
print "TRANSLATERS BUILT"

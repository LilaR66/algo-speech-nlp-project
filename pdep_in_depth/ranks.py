import pickle
import os
import pandas as pd
from collections import Counter

from bssp.common import paths
from bssp.common.config import Config

cfg = Config(
    "clres",
    embedding_model="bert-base-cased",
    override_weights_path=None,
    metric="cosine",
    top_n=50,
    query_n=1,
    bert_layers=[7],
)

df = pd.read_csv(paths.predictions_tsv_path(cfg), sep="\t", encoding = 'unicode_escape', engine ='python')
dfranks = pd.read_csv("cache/clres_stats/ranks.tsv", sep="\t")
dfscores = pd.read_csv("cache/clres_stats/scores.tsv", sep="\t")

def sense_anal(df):
    """
    Identifies the senses in 'label' field of 'df' and counts the number of
    instances in 'df' returning the senses and the counts.
    """
    senses = set()
    scnt = Counter()
    total = 0
    for _, row in df.iterrows():
        label = row.label
        source = label
        source = source[source.index(":")+1:]
        label = label[:label.index(":")]
        senses.add(label)
        scnt[label] += 1
        total += 1
    print("Total Count: ", total)
    return senses, scnt

def prep_anal(df,scnt):
    """
    Identifies the prepositions in the dataframe, with the list of senses
    and the number of test queries with each sense. Requires the sense
    counts from sense_anal
    """
    result = {}
    lemmas = df['lemma'].unique()
    for i in lemmas:
        result[i] = {}
    for i in scnt:
        upd = {}
        prep = i[:i.index("_")]
        sense = i[i.index("_")+1:]
        value = scnt[i]
        upd[sense] = value
        result[prep].update(upd)
    return result
    
def getprep(prep):
    tsv = paths.predictions_tsv_path(cfg)
    tsv = tsv[: tsv.rfind("/")+1] + prep + "-" + tsv[tsv.rfind("/")+1:]
    return pd.read_csv(tsv, sep="\t", encoding = 'unicode_escape', engine ='python')

def corpanal(df):
    preds = Counter()
    for k in range(0,len(df)):
        label_1 = df.loc[k]["label_1"]
        if("OEC" in label_1):
            preds["OEC"] += 1
        elif("FN" in label_1):
            preds["FN"] += 1
        else:
            preds["CPA"] += 1
    print(preds)
    
def corp2(df):
    import re
    test = Counter()
    preds = Counter()
    correct = Counter()
    for _, row in df.iterrows():
        label = row.label
        tcorp = re.search(":([A-Z]*):",label).group(1)
        test[tcorp] += 1
        label = label[:label.index(":")]
        label_1 = row.label_1
        corp = re.search("#([A-Z]*)-",label_1).group(1)
        preds[corp] += 1
        label_1 = label_1[:label_1.index("#")]
        if(label_1 == label):
            correct[corp] += 1
    print("Test:      ", test)
    print("Instances: ", preds)
    print("Correct:   ", correct)
    return test, preds, correct

def dfsanal(dfscores):
    fhi = 0
    ohi = 0
    fhcnt = 0
    flo = 0
    olo = 0
    flcnt = 0
    for _, row in dfscores.iterrows():
        if row.freq >= 50:
            fhi += row.cover
            ohi += row.opt
            fhcnt += 1
        else:
            flo += row.cover
            olo += row.opt
            flcnt += 1
    print("High freq: ", fhcnt, round(fhi/fhcnt,2), "High opt:", round(ohi/fhcnt,3))
    print("Low freq:  ", flcnt, round(flo/flcnt,2), "Low opt: ", round(olo/flcnt,3))
            
def nearest(df):
    testlab = Counter() # the number of test queries for the 'freq'
    near = Counter() # those for which label_1 is the same as the test
    for _, row in df.iterrows():
        freq = row.label_freq_in_train
        if freq <= 50: # used > 50 for the low freq
            continue
        label = row.label
        label = label[:label.index(":")]
        testlab[label] += 1
        label_1 = row.label_1
        label_1 = label_1[:label_1.index("#")]
        if label == label_1:
            near[label] += 1
    return testlab, near

def rank(df):
    dfranks = pd.DataFrame(columns = ['sense', 'freq', 'found', 'score', 'opt', 'label', 'train'])
    testq = Counter() # the number of test queries with each sense (label)
    freqs = Counter() # the number of the training data for the label 
    found = Counter() # the addition for each row for each test query
    scores = Counter() # the scores for the test query
    opts = Counter() # the optimum possibility for the test query
    for _, row in df.iterrows():
        freq = row.label_freq_in_train
        tfreq = freq
        label = row.label
        label = label[:label.index(":")]
        if row.lemma == 'circa': # this doesn't have 50 predictions
            continue
        if freq > 1000:
            continue
        if freq > 50:
            freq = 50
        testq[label] += 1 # how many with this label
        freqs[label] += freq # adding this for each label
        ranks = [] # for which the predictions for this row
        opt = (freq * (freq + 1))/2 # optimum score
        score = 0
        for i in range(50):
            lab = getattr(row, f"label_{i+1}")
            lab = lab[:lab.index("#")]
            if lab == label:
                ranks.append(i+1)
                if i < freq:
                    score += freq - i
        found[label] += len(ranks)
        scores[label] += score
        opts[label] += opt
        dfranks.loc[len(dfranks.index)] = [label, freq, len(ranks), score, int(opt), row.label, tfreq]
    dfscores = pd.DataFrame(columns = ['sense', 'tasks', 'freq', 'in50', 'cover', 'score', 'opt'])
    for i in testq:
        tasks = testq[i]
        pct = scores[i]/opts[i]
        in50 = found[i]/freqs[i]
        dfscores.loc[len(dfscores.index)] = [i, tasks, int(freqs[i]/testq[i]),
                                             str(found[i])+'/'+str(freqs[i]), round(in50,2),
                                             str(scores[i])+'/'+str(int(opts[i])), round(pct,3)] # round(pct,3)]
    return dfranks, dfscores

def avescore(df,sense):
    """
    Creates a dataframe containing the average distances for each
    corpus instance having the sense in the test query
    
    The columns are:
        corp  : the corpus instances
        occ   : the number of test queries containing the corpus instance
        locs  : the sum of the label numbers of the corpus instances
        ave   : the average of the locations (locs/occ)
        dists : the average distances for the corpus instances

    Parameters
    ----------
    df : dataframe containing the predictions for each test query
    sense : a string giving a preposition sense, with the preposition,
        an underscore, and the sense number

    Returns
    -------
    dfaves (dataframe)
        containing the colums above

    """
    dfaves = pd.DataFrame(columns = ['corp', 'occ', 'locs', 'ave', 'dists'])
    tr = Counter() # the positions for the corpus instance
    num = Counter() # the occurrences for the corpus instance
    dists = Counter() # the sum of the distances for the occurrences
    for _, row in df.iterrows():
        # remove the corpus instance from the label and goes to the next
        #   row if the lable isn't the desired sense
        label = row.label
        label = label[:label.index(":")]
        if label != sense:
            continue
        # examines the 50 prediction labels and distances 
        for i in range(50):
            lab = getattr(row, f"label_{i+1}")
            dist = getattr(row, f"distance_{i+1}")
            # gets the corpus instance of the label_{i+1}
            corp = lab[lab.index("#"):]
            corp = corp[corp.rfind("#")+1:]
            lab = lab[:lab.index("#")]
            # when we have a match with the sense, we increment
            if lab == label:
                tr[corp] += i + 1
                num[corp] += 1
                dists[corp] += dist
    for i in tr:
        ave = round(tr[i]/num[i],1)
        avedists = round(dists[i]/num[i],3)
        #print(ave, '\t', i, '\t', tr[i], '\t', num[i])
        dfaves.loc[len(dfaves.index)] = [i, num[i], tr[i], ave, avedists]
    return dfaves

def score_low(df):
    # Identifies test queries that have a nearest neighbor equal t0 a 
    #   distance of 0.0, indicating a duplication in different corpora
    zerocnt = 0
    for _, row in df.iterrows():
        dist = row.distance_1
        if dist == 0:
            zerocnt += 1
            freq = row.label_freq_in_train
            if freq > 50:
                amt = "high"
            else:
                amt = "low"
            print(row.label, '\t', row.label_1, '\t', amt)
    print("Zero distances: ", zerocnt)

def corr(df):
    labels = Counter()
    correct = Counter()

    for k in range(0,len(df)):
        #row = df.loc[k]
        label = df.loc[k]["label"]
        label = label[:label.index(":")]
        labels[label] += 1
        
        label_1 = df.loc[k]["label_1"]
        label_1 = label_1[:label_1.index("#")]
        if(label_1 == label):
            correct[label] += 1
        #print("test", label, " predict", label_1)
    #print(labels)
    #print(correct)
    for sense in correct:
        print(sense,str(correct[sense])+"/"+str(labels[sense]))
    #return labels, correct

"""
pdep = {}
pdep['about'] = ['1(1)', '2(1a)']
pdep['about'] = ['about_1(1)', 'about_2(1a)', 'about_3(2)', 'about_3(2)-1', 'about_5(3a)']
"""
def sense_anal(df):
    """
    Identifies the senses in 'label' field of 'df', removing the 
    corpus source and instance number from the field. Counts the number of
    instances in 'df' and prints out the number.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Contains the predictions for each test instance.

    Returns
    -------
    senses : set
        The label for a test instance. There are usual multiple test
        instances for each senses.
    scnt : collections.Counter
        The same set of labels in 'senses', with a count of the number
        of test instances in 'df'.
    """
    senses = set()
    scnt = Counter()
    total = 0
    for _, row in df.iterrows():
        label = row.label
        source = label
        source = source[source.index(":")+1:]
        label = label[:label.index(":")]
        senses.add(label)
        scnt[label] += 1
        total += 1
    print("Total Count: ", total)
    return senses, scnt

def freq_anal(df):
    lowfreq = set()
    hifreq = set()
    for _, row in df.iterrows():
        label = row.label
        label = label[:label.index(":")]
        if row.label_freq_in_train > 50:
            hifreq.add(label)
            continue
        lowfreq.add(label)
    return lowfreq, hifreq
    
"""
for i in lemmas:
    result[i] = {}
count = 0
for i in scnt:
    upd = {}
    #if(count == 20):
        #break
    prep = i[:i.index("_")]
    sense = i[i.index("_")+1:]
    value = scnt[i]
    upd[sense] = value
    result[prep].update(upd)
    #print(i,scnt[i],prep,sense,value)
    #count += 1
result['about'].update({'1(1)': 236})

"""

    
def corp_test(df):
    """
    Identifies the corpus of the nearest prediction for the corpus of
    the label

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    fn : TYPE
        DESCRIPTION.
    cp : TYPE
        DESCRIPTION.
    oe : TYPE
        DESCRIPTION.

    """
    import re
    fn = Counter()
    cp = Counter()
    oe = Counter()
    for _, row in df.iterrows():
        label = row.label
        tcorp = re.search(":([A-Z]*):",label).group(1)
        label_1 = row.label_1
        pcorp = re.search("#([A-Z]*)-",label_1).group(1)
        if(tcorp == "FN"):
            fn[pcorp] += 1
        elif(tcorp == "CPA"):
            cp[pcorp] += 1
        else:
            oe[pcorp] += 1
    print("FN:  ", fn)
    print("CPA: ", cp)
    print("OEC: ", oe)
    return fn, cp, oe
       

def prec(data):
    cnt = 0
    correct = 0.0
    for i in range(1,len(data)+1):
        cnt += 1
        correct += data[i]['label']
    precision = correct/cnt
    return cnt, precision        
            



def cover(df,hi,lo):
    count = 0
    for _, row in df.iterrows():
        if row.cover >= lo and row.cover <= hi:
            count += 1
            #print(row.tasks, '\t', row.freq, '\t', row.cover)
    print("Count: ", lo, hi, count)

def dfsanal(df):
    fhi = 0
    ohi = 0
    fhcnt = 0
    flo = 0
    olo = 0
    flcnt = 0
    for _, row in df.iterrows():
        if row.freq >= 50:
            fhi += row.cover
            ohi += row.opt
            fhcnt += 1
        else:
            flo += row.cover
            olo += row.opt
            flcnt += 1
    print("High freq: ", fhcnt, round(fhi/fhcnt,2), "High opt:", round(ohi/fhcnt,3))
    print("Low freq:  ", flcnt, round(flo/flcnt,2), "Low opt: ", round(olo/flcnt,3))
            
def dfspace(df):
    tasks = 0
    for _, row in df.iterrows():
        if ' ' in row.sense:
            print(row.sense, ' (', row.tasks, ')')
            tasks += row.tasks
    print("Total tasks with spaces: ", tasks)
    
def avescore(df,sense):
    """
    Creates a dataframe containing the average distances for each
    corpus instance having the sense in the test query
    
    The columns are:
        corp  : the corpus instances
        occ   : the number of test queries containing the corpus instance
        locs  : the sum of the label numbers of the corpus instances
        ave   : the average of the locations (locs/occ)
        dists : the average distances for the corpus instances

    Parameters
    ----------
    df : dataframe containing the predictions for each test query
    sense : a string giving a preposition sense, with the preposition,
        an underscore, and the sense number

    Returns
    -------
    dfaves (dataframe)
        containing the colums above

    """
    dfaves = pd.DataFrame(columns = ['corp', 'occ', 'locs', 'ave', 'dists'])
    tr = Counter() # the positions for the corpus instance
    num = Counter() # the occurrences for the corpus instance
    dists = Counter() # the sum of the distances for the occurrences
    for _, row in df.iterrows():
        # remove the corpus instance from the label and goes to the next
        #   row if the lable isn't the desired sense
        label = row.label
        label = label[:label.index(":")]
        if label != sense:
            continue
        # examines the 50 prediction labels and distances 
        for i in range(50):
            lab = getattr(row, f"label_{i+1}")
            dist = getattr(row, f"distance_{i+1}")
            # gets the corpus instance of the label_{i+1}
            corp = lab[lab.index("#"):]
            corp = corp[corp.rfind("#")+1:]
            lab = lab[:lab.index("#")]
            # when we have a match with the sense, we increment
            if lab == label:
                tr[corp] += i + 1
                num[corp] += 1
                dists[corp] += dist
    for i in tr:
        ave = round(tr[i]/num[i],1)
        avedists = round(dists[i]/num[i],3)
        #print(ave, '\t', i, '\t', tr[i], '\t', num[i])
        dfaves.loc[len(dfaves.index)] = [i, num[i], tr[i], ave, avedists]
    return dfaves

def rank1(df):
    for k in range(0,len(df)):
        row = df.loc[k]
        label = df.loc[k]["label"]
        source = label
        source = source[source.index(":")+1:]
        label = label[:label.index(":")]
        
        sum = 0
        for i in range(50):
            lab = getattr(row, f"label_{i+1}")
            lab = lab[:lab.index("#")]
            if(lab == label):
                sum += (50 - i)
                
        print(k, label, sum, source, df.loc[k]["sentence"][:70])
        
def gettrain(prep):
    split = "train"
    pickle_path = paths.dataset_path(cfg, "train", prep)
    if os.path.isfile(pickle_path):
        print(f"Reading split {split} from cache at {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

def traincorp(tr):
    train = Counter()
    for i in range(0,len(tr)):
        row = tr[i]
        train[row['metas']['source']] += 1 
    return train
        
def missing(test,df):
    for i in range(len(test)):
        lab = test[i]["label"].label+':'+test[i]["metas"]["source"]+":"+test[i]["metas"]["id"]
        row = df[df['label'] == lab].index
        if(len(row)):
            continue
        print(row,lab)

"""
"""
dfaves = avescore(df,"about_2(1a)")
dfaves1 = avescore(df,"on_20(10)")

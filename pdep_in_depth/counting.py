import sqlite3
import pandas as pd
from collections import Counter
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# getting the PDEP corpus if not well downloaded by G&S' bhus repo (with the three different corpora within the repo)
conn = sqlite3.connect('../data/pdep/SQL/prepcorp.sqlite')
conn.row_factory = dict_factory
crows = list(conn.execute('SELECT * FROM prepcorp'))

# getting the PDEP definitions
#conn = sqlite3.connect('data/pdep/SQL/prepdefs.sqlite')
conn = sqlite3.connect('../data/pdep/SQL/prepdefs.sqlite')
conn.row_factory = dict_factory
drows = list(conn.execute('SELECT * FROM prepdefs'))

WHITELIST = ['about', 'above', 'across', 'after', 'against', 'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'circa', 'despite', 'during', 'except', 'for', 'from', 'in', 'including', 'inside', 'into', 'near', 'of', 'off', 'on', 'onto', 'over', 'per', 'since', 'than', 'through', 'to', 'toward', 'towards', 'under', 'until', 'unto', 'up', 'upon', 'via', 'with', 'without']
PDEPLIST = ["'cept", "'gainst", "'mongst", "'pon", 'a cut above', 'a la', 'abaft', 'aboard', 'about', 'above', 'absent', 'according to', 'across', 'afore', 'after', 'after the fashion of', 'against', 'agin', 'ahead of', 'all for', 'all over', 'along', 'along with', 'alongside', 'amid', 'amidst', 'among', 'anent', 'anti', 'apart from', 'apropos', 'around', 'as', 'as far as', 'as for', 'as from', 'as of', 'as regards', 'as to', 'as well as', 'aside from', 'aslant', 'astraddle', 'astride', 'at', 'at a range of', 'at the hand of', 'at the hands of', 'at the heels of', 'athwart', 'atop', 'back of', 'bar', 'bare of', 'barring', 'because of', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'betwixt', 'beyond', 'but', 'but for', 'by', 'by courtesy of', 'by dint of', 'by force of', 'by means of', 'by reason of', 'by the hand of', 'by the hands of', 'by the name of', 'by virtue of', 'by way of', 'care of', 'chez', 'circa', 'come', 'complete with', 'concerning', 'considering', 'contrary to', 'counting', 'courtesy of', 'cum', 'dehors', 'depending on', 'despite', 'down', 'due to', 'during', 'ere', 'ex', 'except', 'except for', 'excepting', 'excluding', 'exclusive of', 'failing', 'following', 'for', 'for all', 'for the benefit of', 'forbye', 'fore', 'fornent', 'frae', 'from', 'give or take', 'given', 'gone', 'having regard to', 'in', 'in accord with', 'in addition to', 'in advance of', 'in aid of', 'in back of', 'in bed with', 'in behalf of', 'in case of', 'in common with', 'in company with', 'in connection with', 'in consideration of', 'in contravention of', 'in default of', 'in excess of', 'in face of', 'in favor of', 'in favour of', 'in front of', 'in honor of', 'in honour of', 'in keeping with', 'in lieu of', 'in light of', 'in line with', 'in memoriam', 'in need of', 'in peril of', 'in place of', 'in proportion to', 'in re', 'in reference to', 'in regard to', 'in relation to', 'in respect of', 'in sight of', 'in spite of', 'in terms of', 'in the course of', 'in the face of', 'in the fashion of', 'in the grip of', 'in the light of', 'in the matter of', 'in the midst of', 'in the name of', 'in the pay of', 'in the person of', 'in the shape of', 'in the teeth of', 'in the throes of', 'in token of', 'in view of', 'in virtue of', 'including', 'inclusive of', 'inside', 'inside of', 'instead of', 'into', 'irrespective of', 'less', 'like', 'little short of', 'mid', 'midst', 'minus', 'mod', 'modulo', 'more like', 'near', 'near to', 'neath', 'next door to', 'next to', 'nigh', 'nothing short of', 'notwithstanding', "o'", "o'er", 'of', 'of the name of', 'of the order of', 'off', 'on', 'on a level with', 'on a par with', 'on account of', 'on behalf of', 'on pain of', 'on the order of', 'on the part of', 'on the point of', 'on the score of', 'on the strength of', 'on the stroke of', 'on top of', 'onto', 'opposite', 'other than', 'out of', 'out of keeping with', 'out of line with', 'outboard of', 'outside', 'outside of', 'outta', 'outwith', 'over', 'over against', 'over and above', 'overtop', 'owing to', 'pace', 'past', 'pending', 'per', 'plus', 'preparatory to', 'previous to', 'prior to', 'pro', 'pursuant to', 'qua', 're', 'regarding', 'regardless of', 'relative to', 'respecting', 'round', 'round about', 'sans', 'save', 'saving', 'short for', 'short of', 'since', 'subsequent to', 'than', 'thanks to', 'this side of', "thro'", 'through', 'throughout', 'thru', 'thwart', 'till', 'to', 'to the accompaniment of', 'to the tune of', 'together with', 'touching', 'toward', 'towards', 'under', 'under cover of', 'under pain of', 'under sentence of', 'under the heel of', 'underneath', 'unlike', 'until', 'unto', 'up', 'up against', 'up and down', 'up before', 'up for', 'up to', 'upon', 'upside', 'upward of', 'upwards of', 'versus', 'via', 'vice', 'vis-a-vis', 'while', 'with', 'with reference to', 'with regard to', 'with respect to', 'with the exception of', 'within', 'within sight of', 'without']

def pdepinsts(crows):
    # identifying instances for non PDEP prepositions
    count = 0
    nonprep = 0
    nonpreps = set()
    for num_rows_visited, row in enumerate(crows):
        if row['prep'] in PDEPLIST:
            count += 1
        if row['prep'] not in PDEPLIST:
            nonprep += 1
            nonpreps.add(row['prep'])

    print("Instances: " + str(len(crows)), ";", 
          "PDEP: " + str(count), ";",
          "Not Used: " + str(nonprep))
    print("Not Used: " + str(nonpreps))

def unused(crows):
    # counts for the three corpora, primarily identifying instances that are not used
    count = 0
    odd = Counter() # non-prep, adverbs, phrasal verbs
    mwe = Counter() # multiword preposition instances
    non = Counter() # single-word prepositions not used
    off = Counter() # instances with offset character locations
    uncommon = set() # list of single-word prepositions not used
    for num_rows_visited, row in enumerate(crows):
        sense = row['sense']
        source = row['source']
        # Skip odd-looking senses
        if sense in ['unk', 'x', 'pv', 'adverb', '1(!)', '']:
            odd[source] += 1
            continue
        # Skip multiword prepositions like "out of"
        if ' ' in row['prep']:
            mwe[source] += 1
            continue
        if row['inst'] in [577203,]:
            odd[source] += 1
            continue
        # Only keep preps that are relatively common
        if row['prep'].lower() not in WHITELIST:
            non[source] += 1
            uncommon.add(row['prep'])
            continue
        # Index is char-based, so tokenization procedure needs to proceed by first excising the preposition
        prep_offset = row['preploc']
        prep_end = row['sentence'].find(' ', prep_offset+1)
        prep_token = row['sentence'][prep_offset:prep_end].strip()
        # Skip if we don't have a match (some offsets are unreliable)
        if prep_token.lower() != row['prep'].lower():
            off[source] += 1
            continue
        count += 1
    print(count)
    print("Odd: ", odd)
    print("MWE: ", mwe)
    print("Off: ", off)
    print("Non: ", non)
    print("Uncommon: ", uncommon)

def defs(drows):
    # identifying PDEP senses
    count = 0
    defs = set()
    for num_rows_visited, row in enumerate(drows):
        if(num_rows_visited == 1040):
            break
        if(row['prep'] == "apropos of"):
            continue
        sense = row['prep'] + "_" + row['sense']
        defs.add(sense)
        count += 1
    print("PDEP senses: " , str(count))
    return defs

#tsv = "cache\clres_stats\pdep_senses.tsv"
tsv = "..\cache\clres_stats\pdep_senses.tsv"

def with_insts():
    # Read the 'tsv' file into 'insts', iterate through it to create the set
    #   to identify the senses that have instances in PDEP
    used = set()
    senses = pd.read_csv(tsv, sep="\t")
    for _, row in senses.iterrows():
        used.add(row["'sense'"])
    return used

def no_insts(defs,used):
    # identifying PDEP senses that have no corpus instances
    unused = set()
    for i in defs:
        if i in used:
            continue
        unused.add(i)
    return unused

def polyused(used):
    # identifying PDEP senses that have a space in the sense 
    #   also indicating that the tagged instances were viewed
    #   as polysemous
    poly = set()
    for i in used:
        sno = i[i.index("_")+1:]
        if " " in sno:
            poly.add(i)
    return poly

def csenses(crows):
    # adds the sense numbers, appending to the preposition
    # identifying instances with polysemous tags
    csenses = Counter()
    poly = Counter()
    for num_rows_visited, row in enumerate(crows):
        prep = row['prep'].lower()
        if row['sense'] in ['unk', 'x', 'pv', 'adverb', '1(!)', '']:
            continue
        if row['inst'] in [577203,]:
            continue
        sense = prep + '_' + row['sense']
        csenses[sense] += 1
        if ' ' in row['sense']:
            poly[sense] += 1
    return csenses

def mwe(defs):
    # identifies the multiple-word entries and senses in PDEP,
    #   indicating entries needing further analysis
    mwe_ents = set()
    mwe_sens = set()
    for i in defs:
        if " " in i:
            mwe_sens.add(i)
            mwe_ents.add(i[:i.index("_")])
    return mwe_ents, mwe_sens

def uncommon(defs,used,unc):
    # examines the single-word prepositions that have not have not been
    #   analyzed in the G&S paper
    uwds = set() # another set of these words, as in 'unc'
    usen = set() # senses for these prepositions
    ucnt = Counter() # the number of senses for these prepositions
    udefs = 0 # a count of the total number of senses for these prepositions
    for i in defs:
        if " " in i:
            continue
        if i not in used:
            continue
        prep = i[:i.index("_")]
        if prep not in unc:
            continue
        uwds.add(prep)
        usen.add(i)
        ucnt[prep] += 1
        udefs += 1
    return uwds, usen, ucnt, udefs

def nondefs(senses,defs):
    # idenifies 'label' not in the PDEP defs 'senses' from ranks.py,
    #   i.e., non-PDEP senses in the dataframe
    nondefs = set()
    for i in senses:
        if(i not in defs):
            nondefs.add(i)
    return nondefs

# tributary prepositions in PDEP
trib = ["'cept", "'gainst", "'mongst", "'pon", 'afore', 'agin', 'amidst', 'betwixt', 'fore', 'frae', 'neath', 'nigh', "o'", "o'er", 'outta', 'outwith', 'sans', "thro'", 'thru', 'thwart', 'till', 'toward', 'upon', 'while']

def unused_defs(unused,trib):
    # identify which unused definitions have no instances: those with
    #   tributary prepositions and those that have regular prepositions
    #   that have senses with no instances in any of the corpora
    tribdefs = Counter()
    pdepdefs = Counter()

    for i in unused:
        prep = i[:i.index("_")]
        #print(prep)
        if prep in trib:
            tribdefs[prep] += 1
        else:
            pdepdefs[prep] += 1
    return tribdefs, pdepdefs
    
def corps():
    train = '../data/pdep/pdep_train.conllu'
    test = '../data/pdep/pdep_test.conllu'
    te = Counter()
    tr = Counter()
    with open(test) as f:
        for line in f:
            if line.startswith("# source = "):
                line = line.replace("# source = ", "")
                line = line.replace("\n", "")
                te[line] += 1
    with open(train) as f:
        for line in f:
            if line.startswith("# source = "):
                line = line.replace("# source = ", "")
                line = line.replace("\n", "")
                tr[line] += 1
    return te, tr

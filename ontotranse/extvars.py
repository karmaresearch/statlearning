from operator import itemgetter
from collections import namedtuple
import numpy as np
import time
from graphviz import Graph
from numpy import argsort

DetailExtVar = namedtuple('DetailExtVar', 'p ent type')

def _shortenURI(ur):
    if ur.startswith('<http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#'):
        return ur.replace('<http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#', '<lubm#')
    elif ur.startswith('<http://www.w3.org/1999/02/22-rdf-syntax-ns#'):
        return ur.replace('<http://www.w3.org/1999/02/22-rdf-syntax-ns#','<rdf#')
    else:
        return ur

class ParamsPostProcessing:
    def __init__(self, threshold = 0, nvars = 10,
                 minsize = 3, minmatch = 0.5,
                 defaultincr = 10,
                 enablepost2 = False,
                 post2_increment = 10,
                 post2_topk = 10):
        self.threshold = threshold # Number of top-k elements to ignore
        self.nvars = nvars # Number of the first top patterns to consider
        self.minsize = minsize # min number of explicit answers in order to consider patterns
        self.minmatch = minmatch # percentage of explicit answer that must match with the elements of the pattern
        self.defaultincr = defaultincr # How much increment should I add to the scores?
        self.enablepost2 = enablepost2 # Enable broader postprocessing (post2)
        self.post2_increment = post2_increment # How much should I increase the scores during the post2 procedure?
        self.post2_topk = post2_topk # maximum number of frequent variables to use in the post2 procedure


class ExtVars:

    def __init__(self):
        self._extvars = []
        self._instances2extvars = {}
        self.nextvars = 0

        # Used for stats
        self.vars2pred = {}
        self.pred2vars = {}

    def load(self, triples, r2e, dicte, dictr, db, minSize, varSOEnabled=False):
        self.r2e = r2e
        self.dicte = dicte
        self.nentities = len(self.dicte)
        self.dictr = dictr
        self.db = db

        # Get the ID of the relation type
        #typeID = np.inf
        #for k,v in dictr.items():
        #    if v == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        #        typeID = k
        #        break
        #self.typeID = typeID
        #self.typeID_db = db.lookup_id('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>')

        # Sort the triples by predicate, object
        triples = sorted(triples, key=itemgetter(1,2))

        extvars = []
        instances2extvars = {}

        # Go through all the training set: For frequent pairs of (p,o), create an existential variable
        prevIdx = 0
        idx = 0
        current_po = (-1, -1)
        for t in triples:
            if (t[1] != current_po[1] or t[2] != current_po[0]):
                if idx - prevIdx > minSize:
                    # if idx - prevIdx < maxSize:
                        # Add an existential variable
                        newvar = DetailExtVar(p=current_po[0], ent=current_po[1], type='po')
                        extvars.append(newvar)
                        # Annotate all subjects with it
                        start = prevIdx
                        while start < idx:
                            tinst = triples[start]
                            # Add the subject
                            if tinst[0] not in instances2extvars:
                                instances2extvars[tinst[0]] = {}
                            instances2extvars[tinst[0]][len(extvars) - 1] = 0
                            start += 1
                current_po = (t[2], t[1])
                prevIdx = idx
            idx += 1

        self._instances2extvars = instances2extvars
        self._extvars = extvars
        self.nextvars = len(extvars)

        # Create more variables, this time checking all objects which share the same subject
        if varSOEnabled:
            triples = sorted(triples, key=itemgetter(0,2))
            prevIdx = 0
            idx = 0
            current_sp = (-1, -1)
            for t in triples:
                if (t[0] != current_sp[0] or t[2] != current_sp[1]):
                    if idx - prevIdx > minSize:
                        newvar = DetailExtVar(p=current_sp[1], ent=current_sp[0], type='sp')
                        extvars.append(newvar)
                        start = prevIdx
                        while start < idx:
                            tinst = triples[start]
                            # Add the object
                            if tinst[2] not in instances2extvars:
                                instances2extvars[tinst[2]] = {}
                            instances2extvars[tinst[2]][len(extvars) - 1] = 0
                            start += 1
                    current_sp = (t[0], t[2])
                    prevIdx = idx
                idx += 1

        print("Created new %d existential variables" % len(extvars))

    def getNExtVars(self):
        return self.nextvars

    def getPEnt(self, id):
        out = self._extvars[id]
        return out.p, out.ent, out.type

    def isSubjectPartOfExtVar(self, s, var):
        if s not in self._instances2extvars:
            return False
        vars = self._instances2extvars[s]
        if var in vars:
            return True
        else:
            return False

    def _selectVars(self, vars):
        #Each tuple contains: the average number of matches per answer, the number of answers that could have been matched, the total number of answers' connections and the total number of elements of the variable

        # Sort by the fact the predicate is different and number of matches
        sortedList = sorted(vars, key=itemgetter(5,2))
        sortedList = sortedList[::-1]

        # Keep only the ones that can reach multiple answers
        idx = 0
        for s in sortedList:
            if s[2] <= 1 or s[2] != sortedList[0][2]:
                break
            idx += 1
        sortedList = sortedList[:idx]

        # Sort it by the number of connections from the variable. The less the better
        sortedList = sorted(sortedList, key=itemgetter(4))

        return sortedList[0:1]

    def _doesExVarLeadToAnswer(self, key, poskey, p, answers, pvar, ovar, neigh_v):
        # get all neighbours of the answers
        matches = np.zeros(len(answers))
        idxanswer = 0
        allmatches = set()
        allneigha = set()
        for answer in answers:
            if poskey == 1:
                neigh_a = self.db.o_aggr(answer)
            else:
                neigh_a = self.db.s_aggr(answer)
            inter = set(neigh_a).intersection(neigh_v)
            allmatches = allmatches.union(inter)
            allneigha = allneigha.union(neigh_a)
            if len(inter) > 0:
                matches[idxanswer] = 1
            idxanswer += 1
        return np.sum(matches), len(allmatches), len(allneigha), len(neigh_v)

    def _expandTriples(self, possibleVars, key, poskey, currentP, answers, output):
        # Select which variable to add
        selectedVars = []
        for pvar,os in possibleVars.items():
            for ovar in os:
                idvar = ovar.idvar
                medmatches, nmatches, size_a, size_v = self._doesExVarLeadToAnswer(key, poskey, currentP, answers, pvar, ovar.o, ovar.answers)
                selectedVars.append((idvar, medmatches, nmatches, size_a, size_v, pvar != currentP))

        selectedVars = self._selectVars(selectedVars)

        for selectedVar in selectedVars:
            # statistics
            var = selectedVar[0]
            if currentP not in self.pred2vars:
                self.pred2vars[currentP] = {}
            listPreds = self.pred2vars[currentP]
            if var not in listPreds:
                listPreds[var] = 0
            listPreds[var] += 1
            if var not in self.vars2pred:
                self.vars2pred[var] = {}
            listVars = self.vars2pred[var]
            if currentP not in listVars:
                listVars[currentP] = 0
            listVars[currentP] += 1

        if key not in output:
            output[key] = ({}, {}) # pos == 0, pos == 1
        keyset = output[key][poskey]
        keyset[currentP] = (answers, selectedVars)

        # Add in output
        #else:
        #    for selectedVar in selectedVars:
        #       for cs in answers:
        #            if poskey == 1:
        #               output.add((cs, self.nentities + selectedVar[0], currentP))
        #            else: # poskey == 0
        #               output.add((self.nentities + selectedVar[0], cs, currentP))

    def _addSubGraphs(self, key, pos, po, output):
        Pair_O_IDVar = namedtuple("Pair_O_IDVar", "o, idvar, answers")
        possibleSG = {}
        allvars = self._instances2extvars[key].keys()
        for dv in allvars:
            pvar, ent, type = self.getPEnt(dv)
            if type != 'po':
                raise "not implemented"
            if pvar not in possibleSG:
                possibleSG[pvar] = []
            allconnections = self.db.s_aggr(ent)
            possibleSG[pvar].append(Pair_O_IDVar(o=ent, idvar=dv, answers=set(allconnections)))

        if pos == 1:
            currentP = -1
            currentS = []
            for existingpos in po:
                if existingpos[1] != currentP:
                    if currentP != -1:
                       self._expandTriples(possibleSG, key, 1, currentP, currentS, output)
                    currentP = existingpos[1]
                    currentS = []
                currentS.append(existingpos[0])
            if currentP != -1:
                self._expandTriples(possibleSG, key, 1, currentP, currentS, output)
        else:
            currentP = -1
            currentO = []
            for existingpos in po:
                if existingpos[0] != currentP:
                    if currentP != -1:
                       self._expandTriples(possibleSG, key, 0, currentP, currentO, output)
                    currentP = existingpos[0]
                    currentO = []
                currentO.append(existingpos[1])
            if currentP != -1:
                self._expandTriples(possibleSG, key, 0, currentP, currentO, output)

    def enrichTrainingWithExVars(self, input):
        additionaltriples = {}

        # Collecting information about the variables
        sortedBySubj = sorted(input, key=itemgetter(0,2))
        prevS = -1
        po = []
        currentIdx = 0
        timeStart = time.time()
        for t in sortedBySubj:
            if currentIdx % 10000 == 0:
                print("Processed %d records of %d in %0.2f secs." % (currentIdx, len(sortedBySubj), time.time() - timeStart))
            if t[0] != prevS:
                if len(po) > 0: # Check the variables used
                    if prevS in self._instances2extvars:
                        self._addSubGraphs(prevS, 0, po, additionaltriples)
                po = []
                prevS = t[0]
            po.append((t[2], t[1]))
            currentIdx += 1
        if prevS != -1 and len(po) > 0:
           if prevS in self._instances2extvars:
               self._addSubGraphs(prevS, 0, po, additionaltriples)

        sortedByObj = sorted(input, key=itemgetter(1,2))
        prevO = -1
        sp = []
        currentIdx = 0
        for t in sortedByObj:
            if currentIdx % 10000 == 0:
                print("Processed %d records of %d in %0.2f secs." % (currentIdx, len(sortedByObj), time.time() - timeStart))
            if t[1] != prevO:
                if len(sp) > 0:
                    # Check the variables used
                    if prevO in self._instances2extvars:
                       self._addSubGraphs(prevO, 1, sp, additionaltriples)
                sp = []
                prevO = t[1]
            sp.append((t[0], t[2]))
            currentIdx += 1
        if prevO != -1 and len(sp) > 0:
            if prevO in self._instances2extvars:
                self._addSubGraphs(prevO, 1, sp, additionaltriples)

        # Reduce the scope of the variables to only some predicates
        filteredPreds = {}
        for var, predsPerVar in self.vars2pred.items():
            # I exclude all variables which are picked from more than 10 predicates
            if len(predsPerVar) > 10:
                print("Skipped variable %d with count %d" % (var, len(predsPerVar)))
                continue

            # Keep only the first three predicates
            listP = []
            for key, value in predsPerVar.items():
                listP.append((key, value))
            listP = sorted(listP, key=itemgetter(1))
            listP = listP[::-1]
            listP = listP[:5] # Take at most 5 preds per var
            for pair in listP:
                if pair[0] not in filteredPreds:
                    filteredPreds[pair[0]] = []
                filteredPreds[pair[0]].append(var)

        #debug code
        #with open('addedtriples.bin', 'wb') as f:
        #    pickle.dump(additionaltriples, f)
        #with open('varcounts.bin', 'wb') as f:
        #    pickle.dump(self.vars2pred, f)
        #with open('filteredpreds.bin', 'wb') as f:
        #    pickle.dump(filteredPreds, f)

        # Process all the variables
        output = set()
        for kg, vg in additionaltriples.items():
            # vg contains the pos
            for pos in range(2):
                panswers = vg[pos]
                for p, answers in panswers.items():
                    vars = []
                    for var in answers[1]:
                        if p in filteredPreds and var[0] in filteredPreds[p]:
                            vars.append(var[0])
                    for ans in answers[0]:
                        for var in vars:
                            if pos == 0:
                                pvar, ovar, tvar = self.getPEnt(var)
                                if p == pvar and ans == ovar:
                                    continue
                                output.add((self.nentities + var, ans, p))
                            else:
                                output.add((ans, self.nentities + var, p))

        print("The extvars added %d new triples to the training set" % len(output))

        #with(open('output.bin', 'wb')) as f:
        #    pickle.dump(output, f)


        # Print statistics
        #for var, ps in self.vars2pred.items():
        #    listp = []
        #    for k,v in ps.items():
        #        listp.append((k,v))
        #    listp = sorted(listp, key=itemgetter(1))
        #    listp = listp[::-1]
        #    print("Variable %d was connected to %d preds %s" % (var, len(ps), str(listp[:10])))
        return list(output)

    def postProcessingStats(self, mdl, scorer, inverse, threshold, question, p, answer, db, varpos, idxscores, scores):
        p_db = db.lookup_id(self.dictr[p])
        if inverse:
            existingResults = db.s(p_db, question)
        else:
            existingResults = db.o(question, p_db)
        existingResults = set(existingResults)
        topresults = set(idxscores[:threshold])

        # Used to propagate some debugging messages
        msg = ''
        count = 0
        maxincr = 10
        for i in range(10):
            count += 1
            t = self.getPEnt(varpos[i] - self.nentities)
            p_db = self.r2e[t[0]]
            lels = db.s(p_db, t[1])
            els = set(lels)
            subjectsToIncrease = els.difference(topresults)
            subjectsToIncrease = subjectsToIncrease.difference(existingResults)
            if len(subjectsToIncrease) > 0 and len(existingResults) > 3:
                match = len(els) - len(subjectsToIncrease) # number of elements which are either in the top-k or in the existing set of answers
                if match > 0:
                    incr = 10 / (len(els) / match)
                    if incr > maxincr:
                        maxincr = incr
                    scores_var = scorer(mdl, t[1], t[0]).flatten()
                    sortidx_var = argsort(scores_var)[::-1]
                    # Remove all the embeddings that corresponds to variables
                    idxvarpos_var = np.where(sortidx_var >= self.nentities)
                    sortidx_var = np.delete(sortidx_var, idxvarpos_var)
                    subjectsToIncrease = sortidx_var[:len(subjectsToIncrease) * 2]
                    scores[list(subjectsToIncrease)] += incr

        # Make sure the top results stay up
        scores[idxscores[:threshold]] += count * maxincr
        scores = scores[:self.nentities]
        newsortidx = argsort(scores)[::-1]
        newPosHead = np.where(newsortidx == answer)[0][0] + 1
        return newPosHead, scores, msg

    def postProcessing1(self, params, posHead, inverse, question, p, answer, db, idxvarpos, varpos, idxscores, scores):
        # Init params
        threshold = params.threshold
        nvars = params.nvars
        minsize = params.minsize
        minmatch = params.minmatch
        defaultincr = params.defaultincr
        # End init params

        # Get existing resuls
        msg = ''
        p_db = db.lookup_id(self.dictr[p])
        if inverse:
            existingResults = db.s(p_db, question)
        else:
            existingResults = db.o(question, p_db)
        if len(existingResults) < minsize:
            return posHead, scores, msg

        existingResults = set(existingResults)
        topresults = set(idxscores[:threshold])
        # Used to propagate some debugging messages
        maxincr = 0
        count = 0
        varsThatContributed = []
        for i in range(nvars):
            t = self.getPEnt(varpos[i] - self.nentities)
            p_db = self.r2e[t[0]]
            lels = db.s(p_db, t[1])
            els = set(lels)
            subjectsToIncrease = els.difference(topresults)
            subjectsToIncrease = subjectsToIncrease.difference(existingResults)
            if len(subjectsToIncrease) > 0:
                    match = len(els) - len(subjectsToIncrease) # number of elements which are either in the top-k or in the existing set of answers
                    if match >= len(existingResults) * minmatch:
                        incr = defaultincr / (len(els) / match)
                        if incr > maxincr:
                            maxincr = incr
                        scores[list(subjectsToIncrease)] += incr
                        count += 1
                        varsThatContributed.append((t[0], t[1]))
        # Make sure the top results stay up
        scores[idxscores[:threshold]] += count * maxincr

        scores = scores[:self.nentities]
        newsortidx = argsort(scores)[::-1]
        newPosHead = np.where(newsortidx == answer)[0][0] + 1

        #debug
        #firstvar = -1
        #if len(idxvarpos[0]) > 0:
        #    firstvar = idxvarpos[0][0]
        #po_list = db.allpo(question)
        #ps_list = db.allps(question)
        #if posHead < newPosHead:
        #    # Check the number of connections of the question
        #    print("pred=%d      BAD %d=>%d degree=%d/%d idxvars=%d vars=%s" % (p, posHead, newPosHead, len(po_list),
        #                                                                len(ps_list), firstvar,
        #                                                                str(varsThatContributed)))
        #elif posHead > newPosHead:
        #    print("pred=%d GOOD %d=>%d degree=%d/%d idxvars=%d vars=%s" % (p, posHead, newPosHead, len(po_list),
        #                                                           len(ps_list), firstvar, str(varsThatContributed)))

        return newPosHead, scores, msg

    def postProcessing2(self, params, cache, inverse, question, p, answer, db,
                        varpos, idxscores, scores):
        #init params
        threshold = params.threshold
        nvars = params.nvars
        minmatch = params.minmatch
        increment = params.post2_increment
        topk = params.post2_topkfreqvars

        p_db = db.lookup_id(self.dictr[p])

        if (question, p) in cache:
            results = cache[(question, p)]
        else:
            # Check whether there are existential variables in the first 20 positions
            vars = varpos[:nvars]

            # Collect all the objects the question is connected to
            mostfrequentpairs_query = {}
            po_query = db.po(question)
            for pair in po_query:
                p_p = pair[0]
                o_p = pair[1]
                if o_p not in mostfrequentpairs_query:
                    mostfrequentpairs_query[o_p] = []
                mostfrequentpairs_query[o_p].append(p_p)

            # Collect the most frequent p,o pairs that connect the object of the query and the object of the existential variables
            mostfrequentpairs = {}
            for i in range(len(vars)):
                v = vars[i]
                if v < self.nentities + self.nextvars:
                    pvar, ovar, tvar = self.getPEnt(v - self.nentities)

                    if tvar == 'sp':
                        raise "Don't know how to process these variables"
                    po = db.po(ovar)
                    for pair in po:
                        p_p = pair[0]
                        o_p = pair[1]
                        if o_p not in mostfrequentpairs:
                            mostfrequentpairs[o_p] = {}
                        if i not in mostfrequentpairs[o_p]:
                            mostfrequentpairs[o_p][i] = []
                        mostfrequentpairs[o_p][i].append(p_p)

            # Calculate a threshold to determine when there is a shared object
            threshold = np.max([int(len(vars) * minmatch), 3])
            freqExtVars = []
            for k,v in mostfrequentpairs.items():
                if len(v) > threshold:
                    freqExtVars.append((k, v, len(v)))

            if len(freqExtVars) > 0:
                # Sort the variables by frequency. Only keep the top 10
                freqExtVars = sorted(freqExtVars, key=itemgetter(2))
                freqExtVars = freqExtVars[0:topk]
            results = []
            for freqObject, freqPreds, lenFreqPreds in freqExtVars:
                allpreds = set()
                for k,vs in freqPreds.items():
                    for v in vs:
                        allpreds.add(v)
                # I assume the majority of the exvars with the same objects had the same predicate
                idvar = list(freqPreds.keys())[0]
                pvar, ovar, tvar = self.getPEnt(vars[idvar]- self.nentities)
                idp = db.lookup_id(self.dictr[pvar])

                for freqPred in allpreds:
                    subjectsToIncrease = db.s(freqPred, freqObject)
                    # subjectsToIncrease is an overestimate. For each value,
                    # I'm going to check how many elements are known to be answers of the query
                    # If the majority is not, then I remove it
                    for subjToIncrease in subjectsToIncrease:
                        if inverse:
                            potentialAnswers = db.s(idp, subjToIncrease)
                        else:
                            potentialAnswers = db.o(subjToIncrease, idp)
                        ok = False
                        for ts in potentialAnswers:
                            if inverse:
                                if db.exists(ts, p_db, question):
                                    ok = True
                                    break
                            else:
                                if db.exists(question, p_db, ts):
                                    ok = True
                                    break
                        if ok:
                            results += potentialAnswers
            cache[(question, p)] = results

        # boost the results up
        if len(results) > 0:
            results = sorted(results)
            results = np.unique(results)
            if inverse:
                existingResults = db.s(p_db, question)
            else:
                existingResults = db.o(question, p_db)
            results = set(results).difference(set(existingResults))
            results.difference(idxscores[:threshold])
            scores[list(results)] += increment
            scores[idxscores[:threshold]] += increment #Making sure I don't touch the top-threshold values

        #ev.showHeatMap(db, question, answer, results1, results, vars, self.nentities)
        scores = scores[:self.nentities]
        newsortidx = argsort(scores)[::-1]
        newPosHead = np.where(newsortidx == answer)[0][0] + 1
        return newPosHead, scores

    def showHeatMap(self, db, question, answer, resultsToBoost1, resultsToBoost2, vars, nentities, levelsToDraw=2):
        # Extract a subgraph
        idcount = 0
        node2id = {}
        id2node = {}
        #Add the question
        node2id[question] = idcount
        id2node[idcount] = question
        idcount += 1
        #Add the answer
        node2id[answer] = idcount
        id2node[idcount] = answer
        idcount += 1

        edges = set()
        level = 0
        queries = [ question ]
        while level < levelsToDraw and len(queries) > 0:
            newqueries = set()
            for q in queries:
                idq = node2id[q]
                results = db.ps(q)
                for r in results:
                    n = r[1]
                    if n not in node2id:
                        node2id[n] = idcount
                        id2node[idcount] = n
                        idcount += 1
                    idnode = node2id[n]
                    edges.add((idnode, idq))
                    newqueries.add(n)
            queries = newqueries
            level += 1

        # TODO Add all the connections on the last level

        # Add all the variables
        varNodes = set()
        for idvar in vars:
            pvar, ovar, tvar = self.getPEnt(idvar- nentities)
            pvar_db = db.lookup_id(self.dictr[pvar])
            instancesVar = db.s(pvar_db, ovar)
            # Add the object
            if ovar not in node2id:
                node2id[ovar] = idcount
                id2node[idcount] = ovar
                idcount += 1
            idnode = node2id[ovar]
            varNodes.add(idnode)


        #Render the graph
        dot = Graph(comment='')
        for id, nodeLabel in id2node.items():
            c = (1 << 24) - 1
            label=''
            height = '0.3cm'
            width = '0.3cm'

            if id in varNodes:
                label='V'
            if nodeLabel in resultsToBoost1:
                c -= 127 << 8
                c -= 127
            if nodeLabel in resultsToBoost2:
                c -= 127 << 8
                c -= 127
            if nodeLabel in resultsToBoost1 and nodeLabel in resultsToBoost2:
                c -= 127 << 16
            if id == 0: # question node
                label = 'Q'
                height='0.7cm'
                width='0.7cm'
            if id == 1: # answer node
                label='A'
                height='0.7cm'
                width='0.7cm'

            c = '#' + hex(c)[2:]
            dot.node(str(id), label=label, fillcolor=c, style='filled', height=height, width=width)
        for edg in edges:
            dot.edge(str(edg[0]), str(edg[1]))
        dot.render(view=True)

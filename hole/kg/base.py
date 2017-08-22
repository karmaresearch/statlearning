from __future__ import print_function
import argparse
import numpy as np
from numpy import argsort
from collections import defaultdict as ddict
import pickle
import timeit
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

import extvars
import ontology
import time
import random
from operator import itemgetter

from skge import sample
from skge.util import to_tensor

np.random.seed(42)

class Experiment(object):

    def enrichTrainingWithExVars(self, xs, nEntities):
        onto = ontology.getOntology()
        typeID = onto.getTypeID()

        additions = []
        keys = onto.getExVarsPerKey()
        for triple in xs:
            pred = triple[2]
            obj = triple[1]
            if pred == typeID and obj in keys:
                additions.append((triple[0], nEntities + keys[obj][0].id, keys[obj][0].prop))
        print("The ontological extvars added %d new triples to the training set" % len(additions))
        return xs + additions

    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='Knowledge Graph experiment', conflict_handler='resolve')
        self.parser.add_argument('--margin', type=float, help='Margin for loss function')
        self.parser.add_argument('--inite', type=str, default='nunif', help='Initialization method (for E)')
        self.parser.add_argument('--initr', type=str, default='nunif', help='Initialization method (for R)')
        self.parser.add_argument('--lr', type=float, help='Learning rate')
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs')
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches')
        self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)
        self.parser.add_argument('--fin', type=str, help='Path to input data', default=None)
        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=10)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)
        self.parser.add_argument('--mode', type=str, default='rank')
        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.neval = -1
        self.best_valid_score = -1.0
        self.exectimes = []
        self.evActive = False

    def run(self):
        # parse comandline arguments
        self.args = self.parser.parse_args()

        if self.args.mode == 'rank':
            self.callback = self.ranking_callback
        elif self.args.mode == 'lp':
            self.callback = self.lp_callback
            self.evaluator = LinkPredictionEval
        else:
            raise ValueError('Unknown experiment mode (%s)' % self.args.mode)
        self.train()

    def ranking_callback(self, trn, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - trn.epoch_start
        self.exectimes.append(elapsed)
        if self.args.no_pairwise:
            self.log.info("[%3d] time = %ds, loss = %f" % (trn.epoch, elapsed, trn.loss))
        else:
            self.log.info("[%3d] time = %ds, violations = %d" % (trn.epoch, elapsed, trn.nviolations))

        # if we improved the validation error, store model and calc test error
        if (trn.epoch % self.args.test_all == 0) or with_eval:
            # Save a temporary copy of the model
            if self.args.fout is not None:
                st = {
                    'model': trn.model,
                    'ev': self.ev
                }
                fileModel = self.args.fout + '.' + str(trn.epoch)
                with open(fileModel, 'wb') as fout:
                    pickle.dump(st, fout, protocol=2)

            if self.args.onlytraining:
                return True

            # I don't do postprocessing here
            pos_v, fpos_v = self.ev_valid.positions(trn.model, False, self.ev, self.db)
            fmrr_valid = self.ranking_scores(pos_v, fpos_v, trn.epoch, 'VALID')
            self.log.debug("FMRR valid = %f, best = %f" % (fmrr_valid, self.best_valid_score))
            if fmrr_valid > self.best_valid_score:
                self.best_valid_score = fmrr_valid
                #pos_t, fpos_t = self.ev_test.positions(trn.model, enableOntoExtVars, enableExtVars, self.ev, self.db)
                #self.ranking_scores(pos_t, fpos_t, trn.epoch, 'TEST')

                if self.args.fout is not None:
                    st = {
                        'model': trn.model,
                        #'pos test': pos_t,
                        #'fpos test': fpos_t,
                        'pos valid': pos_v,
                        'fpos valid': fpos_v,
                        'exectimes': self.exectimes,
                        'ev': self.ev
                    }
                    with open(self.args.fout, 'wb') as fout:
                        pickle.dump(st, fout, protocol=2)
        return True

    def lp_callback(self, m, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - m.epoch_start
        self.exectimes.append(elapsed)
        if self.args.no_pairwise:
            self.log.info("[%3d] time = %ds, loss = %d" % (m.epoch, elapsed, m.loss))
        else:
            self.log.info("[%3d] time = %ds, violations = %d" % (m.epoch, elapsed, m.nviolations))

        # if we improved the validation error, store model and calc test error
        if (m.epoch % self.args.test_all == 0) or with_eval:
            auc_valid, roc_valid = self.ev_valid.scores(m)

            self.log.debug("AUC PR valid = %f, best = %f" % (auc_valid, self.best_valid_score))
            if auc_valid > self.best_valid_score:
                self.best_valid_score = auc_valid
                auc_test, roc_test = self.ev_test.scores(m)
                self.log.debug("AUC PR test = %f, AUC ROC test = %f" % (auc_test, roc_test))

                if self.args.fout is not None:
                    st = {
                        'model': m,
                        'auc pr test': auc_test,
                        'auc pr valid': auc_valid,
                        'auc roc test': roc_test,
                        'auc roc valid': roc_valid,
                        'exectimes': self.exectimes
                    }
                    with open(self.args.fout, 'wb') as fout:
                        pickle.dump(st, fout, protocol=2)
        return True

    def train(self):
        # read data
        with open(self.args.fin, 'rb') as fin:
            data = pickle.load(fin)
        self.logger.setInput(data)

        self.dicte = data['entities']
        self.dictr = data['relations']
        N = len(data['entities'])
        M = len(data['relations'])
        sz = (N, N, M)
        xs_orig = data['train_subs']

        # Calculate new existential variables
        # Enrich the training with existential variables from the graph
        if self.ev == None:
            self.ev = extvars.ExtVars()
        if self.evActive:
            self.ev.load(xs_orig, data['r2e'], self.dicte, self.dictr, self.db, self.minSize)
            xs = xs_orig + self.ev.enrichTrainingWithExVars(xs_orig)
        else:
            xs = xs_orig
        ys = np.ones(len(xs))

        true_triples = data['train_subs'] + data['test_subs'] + data['valid_subs']
        if self.args.mode == 'rank':
            self.ev_test = self.evaluator(self.log, data['test_subs'], true_triples, N, self.ev.getNExtVars(), self.ev, self.dictr, self.dicte, None, 1.0, self.neval)
            self.ev_valid = self.evaluator(self.log, data['valid_subs'], true_triples, N, self.ev.getNExtVars(), self.ev, self.dictr, self.dicte, None, self.args.valid_sample, self.neval)
        elif self.args.mode == 'lp':
            self.ev_test = self.evaluator(data['test_subs'], data['test_labels'])
            self.ev_valid = self.evaluator(data['valid_subs'], data['valid_labels'])

        # create sampling objects
        if self.args.sampler == 'corrupted':
            # create type index, here it is ok to use the whole data
            sampler = sample.CorruptedSampler(self.args.ne, xs, ti)
        elif self.args.sampler == 'random-mode':
            sampler = sample.RandomModeSampler(self.args.ne, [0, 1], xs, sz)
        elif self.args.sampler == 'subgraph':
            sampler = sample.SubGraphsSampler(self.dicte, self.args.ne, N, self.ev.getNExtVars(), [0, 1], xs, sz, self.ev, self.db, data["r2e"])
        elif self.args.sampler == 'lcwa':
            sampler = sample.LCWASampler(self.args.ne, [0, 1, 2], xs, sz)
        else:
            raise ValueError('Unknown sampler (%s)' % self.args.sampler)

        trn = self.setup_trainer(sz, sampler, self.ev, self.existing_model)
        self.log.info("Fitting model %s with trainer %s and parameters %s" % (
            trn.model.__class__.__name__,
            trn.__class__.__name__,
            self.args)
        )
        trn.fit(xs, ys)
        self.callback(trn, with_eval=True)

    def ranking_scores(self, pos, fpos, epoch, txt):
        hpos = [p for k in pos.keys() for p in pos[k]['head']]
        tpos = [p for k in pos.keys() for p in pos[k]['tail']]
        fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
        ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]

        for p, v in pos.items():
            self.log.debug("->Predicate: %d (%s)" % (p, self.dictr[p]))
            textp = self.dictr[p]
            dbp = self.db.lookup_id(textp)
            if v['head']:
                mrr, mean_pos, hits = compute_scores(np.array(v['head']), calculatenorm=True, db=self.db, pred=dbp, headTail=True)
                self.log.debug("-->HEAD %s: MRR = %.2f, Mean Rank = %.2f, Hits@10 = %.2f, #tests = %d" % (txt, mrr,  mean_pos, hits, len(v['head'])))
                self.logger.addPredicatesEpochResults(txt, 'HEAD', epoch, p, self.dictr[p], mrr, hits, mean_pos, len(v['head']))
            if v['tail']:
                mrr, mean_pos, hits = compute_scores(np.array(v['tail']), calculatenorm=True, db=self.db, pred=dbp, headTail=False)
                self.log.debug("-->TAIL %s: MRR = %.2f, Mean Rank = %.2f, Hits@10 = %.2f, #tests = %d" % (txt, mrr,  mean_pos, hits, len(v['tail'])))
                self.logger.addPredicatesEpochResults(txt, 'TAIL', epoch, p, self.dictr[p], mrr, hits, mean_pos, len(v['tail']))

        fmrr = _print_pos(
            self.logger,
            np.array(hpos + tpos),
            np.array(fhpos + ftpos),
            epoch, txt)
        return fmrr

class FilteredRankingEval(object):

    def __init__(self, logger, xs, true_triples, nentities, nextvars, ev, dictr, dicte, paramsPostProcessing, sampleTests, neval=-1):
        idx = ddict(list)
        tt = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        self.neval = neval
        self.log = logger

        self.nentities = nentities
        self.nextvars = nextvars
        self.dictr = dictr
        self.dicte = dicte
        self.paramsPostProcessing = paramsPostProcessing
        self.ev = ev
        self.sampleTests = sampleTests

        self.sz = len(xs)
        for s, o, p in xs:
            idx[p].append((s, o))

        for s, o, p in true_triples:
            tt[p]['os'][s].append(o)
            tt[p]['ss'][o].append(s)

        self.idx = dict(idx)
        self.tt = dict(tt)

        self.neval = {}
        for p, sos in self.idx.items():
            if neval == -1:
                self.neval[p] = -1
            else:
                self.neval[p] = np.int(np.ceil(neval * len(sos) / len(xs)))

    def join(self, a, b):
        # go through a,b and do a merge join
        out = []
        idx_a = 0
        idx_b = 0
        while idx_a < len(a) and idx_b < len(b):
            t_a = a[idx_a]
            v_b = b[idx_b]
            if t_a[1] == v_b:
                out.append(t_a[0])
                idx_a += 1
            elif t_a[1] > v_b:
                idx_b += 1
            else:
                idx_a += 1
        return out

    def positions(self, mdl, enableExVars=False, ev=None, db=None):
        pos = {}
        fpos = {}

        if hasattr(self, 'prepare_global'):
            self.prepare_global(mdl)

        count = 0
        good = 0
        bad = 0

        timeTotal = time.time()
        npreds = 0
        for p, sos in self.idx.items():
            npreds += 1
            ppos = {'head': [], 'tail': []}
            pfpos = {'head': [], 'tail': []}
            cache = {}

            if hasattr(self, 'prepare'):
                self.prepare(mdl, p)

            timeStart = time.time()
            timePost = 0

            for s, o in sos[:self.neval[p]]:

                if self.sampleTests < 1.0:
                    if random.random() > self.sampleTests:
                        continue

                count += 2
                scores_o = self.scores_o(mdl, s, p).flatten()
                sortidx_o = argsort(scores_o)[::-1]
                # Remove all the embeddings that corresponds to variables
                idxvarpos_o = np.where(sortidx_o >= self.nentities)
                varpos_o = sortidx_o[idxvarpos_o]
                sortidx_o = np.delete(sortidx_o, idxvarpos_o)
                # Get the position of the real object
                posObject = np.where(sortidx_o == o)[0][0] + 1

                if enableExVars and self.paramsPostProcessing != None:
                    startPost = time.time()
                    newPosObject, scores_o, msg = ev.postProcessing1(self.paramsPostProcessing,
                                                                     posObject, False, s, p, o, db,
                                                                     idxvarpos_o, varpos_o, sortidx_o,
                                                                     scores_o)

                    if self.paramsPostProcessing.enablepost2:
                        newPosObject, scores_o = ev.postProcessing2(cache,
                                                                    self.paramsPostProcessing.threshold,
                                                                    False, s, p, o, db, varpos_o,
                                                                    sortidx_o, scores_o)
                    #newPosObject, scores_o, msg = ev.postProcessingStats(mdl, self.scores_s, False, thresholdPostProcessing, s, p, o, db, varpos_o, sortidx_o, scores_o)
                    endPost = time.time()
                    timePost += endPost - startPost
                    if posObject > newPosObject:
                        good += 1
                    elif posObject < newPosObject:
                        bad += 1
                    posObject = newPosObject

                ppos['tail'].append((s, o, posObject))

                rm_idx = self.tt[p]['os'][s]
                rm_idx = [i for i in rm_idx if i != o]
                scores_o[rm_idx] = -np.Inf
                sortidx_o = argsort(scores_o)[::-1]
                idxvarpos_o = np.where(sortidx_o >= self.nentities)
                sortidx_o = np.delete(sortidx_o, idxvarpos_o)
                filteredPosObj = np.where(sortidx_o == o)[0][0]
                pfpos['tail'].append((s, o, filteredPosObj + 1))

                # HEAD PREDICTIONS
                scores_s = self.scores_s(mdl, o, p).flatten()
                sortidx_s = argsort(scores_s)[::-1]
                # Remove all the embeddings that corresponds to variables
                idxvarpos_s = np.where(sortidx_s >= self.nentities)
                varpos_s = sortidx_s[idxvarpos_s]
                sortidx_s = np.delete(sortidx_s, idxvarpos_s)
                # Get the position of the real s
                posHead = np.where(sortidx_s == s)[0][0] + 1

                if enableExVars and self.paramsPostProcessing != None:
                    startPost = time.time()
                    newPosHead, scores_s, msg = ev.postProcessing1(self.paramsPostProcessing,
                                                                   posHead, True, o, p, s, db,
                                                                   idxvarpos_s, varpos_s,
                                                                   sortidx_s, scores_s)

                    if self.paramsPostProcessing.enablepost2:
                        newPosHead, scores_s = ev.postProcessing2(cache,
                                                                    self.paramsPostProcessing.threshold,
                                                                    True, o, p, s, db, varpos_s,
                                                                    sortidx_s, scores_s)

                    #newPosHead, scores_s, msg = ev.postProcessingStats(mdl, self.scores_s, True, thresholdPostProcessing, o, p, s, db, varpos_s, sortidx_s, scores_s)
                    endPost = time.time()
                    timePost += endPost - startPost
                    if posHead < newPosHead:
                        bad += 1
                    elif posHead > newPosHead:
                        good += 1
                    posHead = newPosHead

                ppos['head'].append((s, o, posHead))

                rm_idx = self.tt[p]['ss'][o]
                rm_idx = [i for i in rm_idx if i != s]
                scores_s[rm_idx] = -np.Inf
                sortidx_s = argsort(scores_s)[::-1]
                idxvarpos_s = np.where(sortidx_s >= self.nentities)
                sortidx_s = np.delete(sortidx_s, idxvarpos_s)
                filteredPos = np.where(sortidx_s == s)[0][0] + 1
                pfpos['head'].append((s, o, filteredPos))

            pos[p] = ppos
            fpos[p] = pfpos

        print("Count=%d good=%d bad=%d" % (count, good, bad))

        return pos, fpos

class LinkPredictionEval(object):

    def __init__(self, xs, ys):
        ss, os, ps = list(zip(*xs))
        self.ss = list(ss)
        self.ps = list(ps)
        self.os = list(os)
        self.ys = ys

    def scores(self, mdl):
        scores = mdl._scores(self.ss, self.ps, self.os)
        pr, rc, _ = precision_recall_curve(self.ys, scores)
        roc = roc_auc_score(self.ys, scores)
        return auc(rc, pr), roc

def _print_pos(logger, pos, fpos, epoch, txt):
    mrr, mean_pos, hits = compute_scores(pos)
    fmrr, fmean_pos, fhits = compute_scores(fpos)
    logger.getLog().info(
        "[%3d] %s: MRR = %.2f/%.2f, Mean Rank = %.2f/%.2f, Hits@10 = %.2f/%.2f" %
        (epoch, txt, mrr, fmrr, mean_pos, fmean_pos, hits, fhits)
    )
    logger.addGeneralEpochResults(txt, epoch, mrr, hits, mean_pos)
    return fmrr


def compute_scores(pos, calculatenorm=False, db=None, pred=None, headTail=None, hits=10):
    scores = pos[:,2]
    mrr = np.mean(1.0 / scores)
    mean_pos = np.mean(scores)
    #if not calculatenorm:
    #    nmean_pos = 0
    #else:
    #    for i in range(len(pos)):
    #        t = pos[i]
    #        if headTail:
    #            card = db.ns(pred, t[1])
    #        else:
    #            card = db.no(t[0], pred)
    #        scores[i] -= card
    #        if scores[i] < 0:
    #            scores[i] = 0
    #    nmean_pos = np.mean(scores)
    hits = np.mean(scores <= hits).sum() * 100
    return mrr, mean_pos, hits


def cardinalities(xs, ys, sz):
    T = to_tensor(xs, ys, sz)
    c_head = []
    c_tail = []
    for Ti in T:
        sh = Ti.tocsr().sum(axis=1)
        st = Ti.tocsc().sum(axis=0)
        c_head.append(sh[np.where(sh)].mean())
        c_tail.append(st[np.where(st)].mean())

    cards = {'1-1': [], '1-N': [], 'M-1': [], 'M-N': []}
    for k in range(sz[2]):
        if c_head[k] < 1.5 and c_tail[k] < 1.5:
            cards['1-1'].append(k)
        elif c_head[k] < 1.5:
            cards['1-N'].append(k)
        elif c_tail[k] < 1.5:
            cards['M-1'].append(k)
        else:
            cards['M-N'].append(k)
    return cards

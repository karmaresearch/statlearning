"""
Sampling strategies to generate negative examples from knowledge graphs
with an open-world assumption
"""

from copy import deepcopy
from collections import defaultdict as ddict
from numpy.random import randint


class Sampler(object):

    def __init__(self, n, modes, ntries=100):
        self.n = n
        self.modes = modes
        self.ntries = ntries

    def sample(self, xys):
        res = []
        for x, _ in xys:
            for _ in range(self.n):
                for mode in self.modes:
                    t = self._sample(x, mode)
                    if t is not None:
                        res.append(t)
        return res

class SubGraphsSampler(object):

    def __init__(self, dicte, n, nentities, nvars, modes, xs, sz, ev, db, rel2id, ntries = 100):
        self.dicte = dicte
        self.n = n
        self.nentities = nentities
        self.modes = modes
        self.ntries = ntries
        self.xs = set(xs)
        self.sz = sz
        self.nvars = nvars
        self.db = db
        self.ev = ev
        self.rel2id = rel2id

    def sample(self, xys):
        cleanedres = []
        toMatch = xys
        it = 0

        while len(toMatch) > 0:
            res = []
            for x, _ in toMatch:
                for _ in range(self.n):
                    for mode in self.modes:
                        t = self._sample(x, mode)
                        if t is not None:
                            res.append(t)

            # Check whether the triples are likely to be positive
            stillToMatch = []
            for el in res:
                 # el is a tuple where 0 is the pos of the changed term, 1 is the positive triple and 2 is the negative triple
                mode = el[0]
                post = el[1]
                negt = el[2]
                ok = True

                 # ***** THE CODE COMMENTED BELOW DOES NOT SEEM TO BE EFFECTIVE ******
            #     if post[0] >= self.nentities or post[1] >= self.nentities:
            #         # Check whether the majority of the triples is true, then remove it
            #         if post[mode] >= self.nentities:
            #             # Replace all the variable with its instances and count whether the majority is true
            #             p, ent, typ = self.ev.getPEnt(post[mode] - self.nentities)
            #             if typ == 'po':
            #                 instances = self.db.alls(self.rel2id[p], ent)
            #             else:
            #                 instances = self.db.allo(ent, self.rel2id[p])
            #
            #             if negt[mode] in instances:
            #                 count = 0
            #                 negt2 = list(negt)
            #                 for int in instances:
            #                     negt2[mode] = int
            #                     if tuple(negt2) in self.xs:
            #                         count += 1
            #                         if count > len(instances) / 2:
            #                             break
            #                 if count > len(instances) / 2:
            #                     # The majority of instances that match this variable do have the relation that the instance would like to mark as negative. This is not safe. I add it only when this is not the case.
            #                     ok = False
            #
            #         else:
            #             posvar = 0
            #             if mode == 0:
            #                 posvar = 1
            #             p, ent, typ = self.ev.getPEnt(post[posvar] - self.nentities)
            #             if typ == 'po':
            #                 instances = self.db.alls(self.rel2id[p], ent)
            #             else:
            #                 instances = self.db.allo(ent, self.rel2id[p])
            #             count = 0
            #             negt2 = list(negt)
            #             for int in instances:
            #                 negt2[posvar] = int
            #                 if tuple(negt2) in self.xs:
            #                     count += 1
            #                     if count > len(instances) / 2:
            #                         break
            #             if count > len(instances) / 2:
            #                 # Same as before
            #                 ok = False
            #     #else:
            #         #    if mode == 0:
            #         #        #TODOf Is there any variable defined on p/o?
            #         if mode == 0:
            #             # I replaced the subject
            #             if ev.getVar()
            #
                if ok:
                    cleanedres.append((negt, -1.0))
                else:
                    stillToMatch.append((post, 1.0))
            it += 1
            toMatch = stillToMatch

        return cleanedres

    def _check(self, nex, mode, x):
            #Here I check that the elements that I pick does not contradict any instantiation of the variable in the other position
            if mode == 0:
                mode2 = 1
            else:
                mode2 = 0

            if x[mode2] >= self.nentities:
                pvar, entvar, typevar = self.ev.getPEnt(x[mode2] - self.nentities)
                if typevar == 'po':
                    entities = self.db.alls(self.rel2id[pvar], entvar)
                else:
                    entities = self.db.allo(entvar, self.rel2id[pvar])
                nex2 = list(nex)
                for ent in entities:
                    nex2[mode2] = ent
                    if tuple(nex2) in self.xs:
                        return False

            if tuple(nex) not in self.xs:
                return True
            else:
                return False

    def _sample(self, x, mode):
        nex = list(x)
        res = None
        for _ in range(self.ntries):
            replacement = randint(self.sz[mode])
            nex[mode] = replacement
            if self._check(nex, mode, x):
                res = (mode, x, tuple(nex))
                break
        return res

class RandomModeSampler(Sampler):
    """
    Sample negative triples randomly
    """

    def __init__(self, n, modes, xs, sz):
        super(RandomModeSampler, self).__init__(n, modes)
        self.xs = set(xs)
        self.sz = sz

    def _sample(self, x, mode):
        nex = list(x)
        res = None
        for _ in range(self.ntries):
            replacement = randint(self.sz[mode])
            nex[mode] = replacement
            if tuple(nex) not in self.xs:
                res = (tuple(nex), -1.0)
                break
        return res


class RandomSampler(Sampler):

    def __init__(self, n, xs, sz):
        super(RandomSampler, self).__init__(n)
        self.xs = set(xs)
        self.sz = sz

    def _sample(self, x, mode):
        res = None
        for _ in range(self.ntries):
            nex = (randint(self.sz[0]),
                   randint(self.sz[0]),
                   randint(self.sz[1]))
            if nex not in self.xs:
                res = (nex, -1.0)
                break
        return res


class CorruptedSampler(Sampler):

    def __init__(self, n, xs, type_index):
        super(CorruptedSampler, self).__init__(n)
        self.xs = set(xs)
        self.type_index = type_index

    def _sample(self, x, mode):
        nex = list(deepcopy(x))
        res = None
        for _ in range(self.ntries):
            if mode == 2:
                nex[2] = randint(len(self.type_index))
            else:
                k = x[2]
                n = len(self.type_index[k][mode])
                nex[mode] = self.type_index[k][mode][randint(n)]
            if tuple(nex) not in self.xs:
                res = (tuple(nex), -1.0)
                break
        return res


class LCWASampler(RandomModeSampler):
    """
    Sample negative examples according to the local closed world assumption
    """

    def __init__(self, n, modes, xs, sz):
        super(LCWASampler, self).__init__(n, modes, xs, sz)
        self.counts = ddict(int)
        for s, o, p in xs:
            self.counts[(s, p)] += 1

    def _sample(self, x, mode):
        nex = list(deepcopy(x))
        res = None
        for _ in range(self.ntries):
            nex[mode] = randint(self.sz[mode])
            if self.counts[(nex[0], nex[2])] > 0 and tuple(nex) not in self.xs:
                res = (tuple(nex), -1.0)
                break
        return res


def type_index(xs):
    index = ddict(lambda: {0: set(), 1: set()})
    for i, j, k in xs:
        index[k][0].add(i)
        index[k][1].add(j)
    #for p, idx in index.items():
    #    print(p, len(idx[0]), len(idx[1]))
    return {k: {0: list(v[0]), 1: list(v[1])} for k, v in index.items()}

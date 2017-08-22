from graphviz import Digraph
from bisect import bisect_left
import gzip
from numpy import sqrt, squeeze, zeros_like
from numpy.random import randn, uniform
import numpy as np

global taxo
taxo = None

def loadTaxonomy(inputfile):
    global taxo
    taxo = Taxonomy(inputfile)

def getInstance():
    global taxo
    return taxo

def init_semclasses(sz):
    """
    Initialize the embeddings so that instances of the same class get the same
    vector
    """
    bnd = sqrt(6) / sqrt(sz[0] + sz[1])
    p = uniform(low=-bnd, high=bnd, size=sz)
    emb = squeeze(p)

    tax = getInstance()
    nclasses = tax.getNClasses()
    embclasses = uniform(low=-bnd, high=bnd, size=(nclasses, sz[1]))

    idx = 0
    while (idx < sz[0]):
        classid, found, end = tax.getclass(idx)
        if found:
            while idx < end:
                emb[idx] = embclasses[classid]
                idx += 1
        else:
            idx += 1
    return emb


class Taxonomy:
    edges = {}
    nodes = {}
    ranges = []

    # Private methods
    def _cleanup(self, input):
        if input.startswith('<'):
            input = input[1:-1]
            if input.startswith('http://www.w3.org/2002/07/owl#'):
                input = 'owl_' + input[input.find('#')+1:]
            if input.startswith('http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#'):
                input = 'lubm_' + input[input.find('#')+1:]
            if input.startswith('http://www.w3.org/2000/01/rdf-schema#'):
                input = 'rdfs_' + input[input.find('#')+1:]
            if input.startswith('_:'):
                input = 'blanknode_' + input[2:]
        return input

    # Public methods
    def __init__(self, inputfile):
        rang = False
        subc = False
        # Load the file
        if inputfile.endswith('.gz'):
            fin = gzip.open(inputfile, 'rt')
        else:
            fin = open(inputfile)
        for line in fin:
            if "#Ranges#" in line:
                subc = False
                rang = True
            elif "#Taxonomy#" in line:
                subc = True
                rang = False
            elif rang:
                line = line[:-1]
                tok = line.split('\t')
                self.nodes[tok[0]] = tok[1]
                if len(tok) > 2:
                    self.ranges.append((int(tok[2]), int(tok[3]), int(tok[1])))
            elif subc:
                line = line[:-1]
                tok = line.split('\t')
                source = tok[0]
                dest = tok[1]
                source = self._cleanup(source)
                dest = self._cleanup(dest)
                if source not in self.edges:
                    self.edges[source] = []
                    self.edges[source].append(dest)
        # Sort the ranges
        self.ranges = sorted(self.ranges, key=lambda x: (x == 0, x))
        first, second, third = zip(*self.ranges)
        self.beginRanges = np.asarray(first)
        # Check if the sorting was done correctly
        idx = 0
        prevValue = -1
        while (idx < len(self.ranges)):
            if self.ranges[idx][0] <= prevValue:
                raise ValueError("Sorting is not correct")
            prevValue = self.ranges[idx][0]
            idx += 1

    def getclassesinrange(self, min, max):
        output = []
        begin = min
        for r in self.ranges:
            if r[1] > begin:
                startRange = np.min([r[0], begin])
                endRange = np.min([max, r[1]])
                output.append((r[2], startRange, endRange))
                begin = endRange
            if begin == max:
                break
        if begin < max:
            output.append((np.Inf, begin, max))
        return output

    def getclass(self, idInstance):
        pos = bisect_left(self.beginRanges, idInstance)
        if pos >= len(self.beginRanges):
            return 0, False, 0
        elif idInstance >= self.ranges[pos][0] and idInstance < self.ranges[pos][1]:
            return self.ranges[pos][2], True, self.ranges[pos][1]
        else:
            return 0, False, 0

    def visualize(self):
        # Create the graph
        dot = Digraph(comment='Taxonomy')
        for k, v in self.nodes.items():
            node = self._cleanup(k)
            dot.node(node, label=node + '(' + str(v) + ')')
            for k, v in self.edges.items():
                for dest in v:
                    dot.edge(k, dest)
                    dot.render(view=True)

    def getNClasses(self):
        return len(self.nodes)
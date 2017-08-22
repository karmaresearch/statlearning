import os
import shutil
import pickle
import subprocess

import numpy as np
import scipy.sparse as sp
import argparse
import random
import gzip

def parseNTLine(line):
    # Parse the subject
    if line.startswith('<'):
        s = line[:line.find('> ') + 1]
        line = line[line.find('> ') + 2:]
    else:
        s = line[:line.find(' ')]
        line = line[line.find(' ') + 1:]
    # Parse the predicate
    p = line[:line.find('> ') + 1]
    line = line[line.find('> ') + 2:]
    # Parse the object
    o = line[:line.rfind('.') - 1]
    return (s, p, o)

def convertIntoRDF(inputfile, output):
    for line in open(inputfile, 'rt'):
        line = line[:-1]
        tokens = line.split('\t')
        tokens[0] = '<' + tokens[0] + '>'
        tokens[1] = '<' + tokens[1] + '>'
        tokens[2] = '<' + tokens[2] + '>'
        output.append((tokens[0], tokens[1], tokens[2]))


parser = argparse.ArgumentParser(description='Parser: Generates the data. Needs input graph.')
parser.add_argument('-dp','--datapath', help='Path to the directory which contains the input and will store the output', required=True)
parser.add_argument('-np','--nameprefix', help='Name or prefix of input file name. Eg: "lubm1", "wn18".', required=True)
parser.add_argument('-test','--testthreshold', default=0.01, help='Percentage of the input that should be used for testing', type=float, required=False)
parser.add_argument('-valid','--validthreshold', default=0.01, help='Percentage of the input that should be used for valid', type=float, required=False)
parser.add_argument('-rules', '--rules', default='', help='path to the rules to materialize the input', type=str, required=False)
parser.add_argument('-fmt', '--format', default='rdf', help='format of the input. Can be rdf (default) or FB15K', type=str, required=False)
args = vars(parser.parse_args())

# Input
datapath = args['datapath']
prefix = args['nameprefix']
testThreshold = args['testthreshold']
validThreshold = args['validthreshold']
rules = args['rules']
fmt = args['format']

# Output
idxrelation = 0
relations = {}
invrelations = {}

entities2idx = {}
idx2entities = {}
relations2idx = {}
idx2relations = {}
train = []
valid = []
test = []
train_subs = []
test_subs = []
valid_subs = []

rdfs_uri = "<http://www.w3.org/2000/01/rdf-schema#"
owl_uri = "<http://www.w3.org/2002/07/owl#"
processedTriples = 0
if fmt == 'rdf':
    # Split the array in train, test, valid subsets
    print("Splitting the input in train/test/valid")
    inputpath = datapath + "/%s.nt.gz" % prefix
    for line in gzip.open(inputpath, 'rt'):
        line = line[:-1]
        triple = parseNTLine(line)

        # If the triple contains a term from the RDFS or OWL vocabulary, then it's an ontology and hence should be in
        # the training set
        isOntology = False
        if triple[0].startswith(rdfs_uri) or triple[0].startswith(owl_uri):
            isOntology = True
        elif triple[1].startswith(rdfs_uri) or triple[1].startswith(owl_uri):
            isOntology = True
        elif triple[2].startswith(rdfs_uri) or triple[2].startswith(owl_uri):
            isOntology = True

        if isOntology:
            train.append(triple)
            continue

        idx = random.random()
        if idx < testThreshold:
            test.append(triple)
        elif idx < testThreshold + validThreshold:
            valid.append(triple)
        else:
            train.append(triple)
else:
    print("Converting the input in RDF format ...")
    convertIntoRDF(datapath + "/%s-train.txt" % prefix, train)
    convertIntoRDF(datapath + "/%s-test.txt" % prefix, test)
    convertIntoRDF(datapath + "/%s-valid.txt" % prefix, valid)

# Compress the training file using KOGNAC
print("Dumping the training triples in nt format ...")
trainingNtFile = datapath + "/training.nt.gz"
fout = gzip.open(trainingNtFile, "wt")
for triple in train:
    fout.write("%s %s %s .\n" % triple)
fout.close()
print("Launching KOGNAC to compress the training data")
resp = subprocess.call(['kognac_exec', '-i', trainingNtFile, '-o', datapath, '--serializeTax', datapath + "/taxonomy.gz", '-c', '--sampleArg1', str(128)])
if resp != 0:
    print("Program did NOT terminate correctly!")

# Load the output of KOGNAC into main memory
print("Loading the dictionary of the training set in main memory ...")
dict_train_path = datapath + "/dict.gz"
for line in gzip.open(dict_train_path, 'rt'):
    line = line[:-1]
    idx = int(line[:line.find(' ')])
    line = line[line.find(' ') + 1:]
    length = int(line[:line.find(' ')])
    line = line[line.find(' ') + 1:]
    value = line[:length + 1]
    idx2entities[idx] = value
    entities2idx[value] = idx
print("Loading the training set in main memory ...")
input_train_path = datapath + "/triples.gz"
for line in gzip.open(input_train_path, 'rt'):
    line = line[:-1]
    toks = line.split(' ')
    s = int(toks[0])
    p = int(toks[1])
    o = int(toks[2])
    # Replace the relation with an internal idx
    if p not in relations:
        relations[p] = idxrelation
        invrelations[idxrelation] = p
        p = idxrelation
        idxrelation += 1
    else:
        p = relations[p]
    train_subs.append((s, o, p))

# Create the dictionary of relations
for k,v in relations.items():
    idx2relations[v] = idx2entities[k]
    relations2idx[idx2entities[k]] = v

# Remove from the valid/test datasets all triples that contain terms that never appeared in the training set
print("Create the valid dataset")
for t in valid:
    if t[0] in entities2idx and t[2] in entities2idx and t[1] in relations2idx:
        s = entities2idx[t[0]]
        p = relations2idx[t[1]]
        o = entities2idx[t[2]]
        valid_subs.append((s, o, p))
print("Create the test dataset")
for t in test:
    if t[0] in entities2idx and t[2] in entities2idx and t[1] in relations2idx:
        s = entities2idx[t[0]]
        p = relations2idx[t[1]]
        o = entities2idx[t[2]]
        test_subs.append((s, o, p))
print("Finished creating train/valid/test datasets: Train %d, Valid %d, Test %d triples" % (len(train_subs), len(valid_subs), len(test_subs)))

# Extract the ontology from the input
ontoPreds = {}
ontoPreds2 = {}
ontology = {}
for k,v in relations2idx.items():
    if k == '<http://www.w3.org/2002/07/owl#someValuesFrom>':
        ontoPreds[v] = k
        ontoPreds2[k] = v
    elif k == '<http://www.w3.org/2002/07/owl#onProperty>':
        ontoPreds[v] = k
        ontoPreds2[k] = v
for triple in train_subs:
    if triple[2] in ontoPreds:
        if triple[2] not in ontology:
            ontology[triple[2]] = []
        ontology[triple[2]].append(triple)

# Load the input in VLog
print("Creating a database from the training data")
vlogDatabase = datapath + "/vlog"
shutil.rmtree(vlogDatabase, ignore_errors=True)
resp = subprocess.call(['vlog', 'load', '--comprinput', input_train_path, '--comprdict', dict_train_path, '-o', vlogDatabase])
if resp != 0:
    print("Program did NOT terminate correctly!")
if rules != '':
    print("Materialize the training data ...")
    #Load the enriched training data
    vlogExport = datapath + "/vlog_export"
    edbPath = datapath + "/edb.conf"
    print("Launch reasoning on the training data ...")
    resp = subprocess.call(['vlog', 'mat', '--edb', edbPath, '--rules', rules, '--storemat_path', vlogExport, '--storemat_format', 'nt', '--decompressmat', 'true'])
    if resp != 0:
        print("Program did NOT terminate correctly!")
    inferred = 0
    for file in os.listdir(vlogExport):
        if file.startswith('.'):
            continue
        for line in gzip.open(vlogExport  + '/' + file, 'rt'):
            # Parse the nt line
            tokens = parseNTLine(line)
            s = entities2idx[tokens[0]]
            if tokens[1] not in relations2idx:
                idx2relations[idxrelation] = tokens[1]
                relations2idx[tokens[1]] = idxrelation
                idxrelation += 1
            p = relations2idx[tokens[1]]
            o = entities2idx[tokens[2]]
            train_subs.append((s,o,p))
            inferred += 1
    print("Added to the original training data new %d inferred triples" % inferred)

    print("Re-genetate the database with the materialized triples")
    vlogExport2 = vlogExport + "_2"
    resp = subprocess.call(['vlog', 'mat', '--edb', edbPath, '--rules', rules, '--storemat_path', vlogExport2, '--storemat_format', 'nt', '--decompressmat', 'false'])
    matfile = vlogExport2 + "/out-0.nt.gz"
    writer = gzip.open(matfile, 'at')
    for line in gzip.open(input_train_path, "rt"):
        writer.write(line)
    writer.close()
    shutil.move(vlogDatabase, datapath + "/vlog_nomat")
    # Regenerate the database
    resp = subprocess.call(['vlog', 'load', '--comprinput', matfile, '--comprdict', dict_train_path, '-o', vlogDatabase])
    if resp != 0:
        print("Program did NOT terminate correctly!")
    shutil.rmtree(vlogExport2)

# Remove the nt file of the training set
os.remove(trainingNtFile)

print("Writing the python file ...")
ent = {"entities": idx2entities}
rel = {"relations": idx2relations }
rel2 = {"r2e": invrelations } #Idx relations in r to idx relations in e
trn = {"train_subs": train_subs}
tst = {"test_subs": test_subs}
vld = {"valid_subs": valid_subs}
ont = {"onto": ontology}
ont2 = {"ontopreds": ontoPreds2 }
if rules != '':
    reasoning = {"inferred": True}
else:
    reasoning = {"inferred": False}
h = open(datapath + '%s.bin' % prefix, 'wb')
ent.update(rel)
ent.update(rel2)
ent.update(trn)
ent.update(tst)
ent.update(vld)
ent.update(ont)
ent.update(ont2)
ent.update(reasoning)
pickle.dump(ent, h)
h.close()
print("Done.")

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

inputFile = sys.argv[1]
outputDirBase = inputFile + "_out"
if os.path.exists(outputDirBase):
    shutil.rmtree(outputDirBase)
os.makedirs(outputDirBase)
logs = json.load(open(inputFile))

for typ in ('VALID', 'TEST'):
    outputDir = outputDirBase + "/" + typ
    os.makedirs(outputDir)

    # Calculate data to plot
    x = []
    meanrank = []
    hits = []
    for epoch in logs['epochs']:
        if epoch[0] == 'VALID':
            if len(x) == 0 or x[len(x) - 1] != epoch[1]:
                x.append(epoch[1])
                meanrank.append(epoch[4])
                hits.append(epoch[2])
    width = .6
    xbars = np.arange(len(x))

    # Plot mean rank
    plt.figure()
    plt.bar(xbars, meanrank, width=width)
    plt.title('MeanRank/Epochs')
    plt.xticks(xbars + width / 2, x)
    plt.ylabel('MeanRank')
    plt.xlabel('Epochs')
    plt.savefig(outputDir + '/meanrank_epochs.pdf')
    plt.close()

    # Plot hits@10
    plt.figure()
    plt.bar(xbars, hits, width=width)
    plt.title('Hits/Epochs')
    plt.xticks(xbars + width / 2, x)
    plt.ylabel('Hits@10')
    plt.xlabel('Epochs')
    plt.savefig(outputDir + '/hits_epochs.pdf')
    plt.close()

    # Plot mean rank per predicate
    meanranksHeads = {}
    meanranksTails = {}
    prevEpochH = -1
    prevEpochT = -1
    for k,v in logs['preds'].items():
        meanranksHeads[k] = []
        meanranksTails[k] = []
        for el in v:
            if el[0] == 'VALID':
                if prevEpochH != el[1]:
                    if el[2] == 'HEAD':
                        meanranksHeads[k].append(el[5])
                        prevEpochH = el[1]
                if prevEpochT != el[1]:
                    if el[2] == 'TAIL':
                        meanranksTails[k].append(el[5])
                        prevEpochT = el[1]

    #Plot HEAD predictions
    outputDirHead = outputDir + "/head"
    os.makedirs(outputDirHead)
    for k,v in meanranksHeads.items():
        if len(v) > 0:
            plt.figure()
            plt.bar(xbars, v, width=width)
            plt.title('MeanRank H ' + str(k) + '(' + logs['predsNames'][k] + ')')
            plt.xticks(xbars + width / 2, x)
            plt.ylabel('MeanRank')
            plt.xlabel('Epochs')
            plt.savefig(outputDirHead + '/meanrank_head-' + str(k) + '.pdf')
            plt.close()

    #Plot TAIL predictions
    outputDirTail = outputDir + "/tail"
    os.makedirs(outputDirTail)
    for k,v in meanranksTails.items():
        if len(v) > 0:
            plt.figure()
            plt.bar(xbars, v, width=width)
            plt.title('MeanRank T ' + str(k) + '(' + logs['predsNames'][k] + ')')
            plt.xticks(xbars + width / 2, x)
            plt.ylabel('MeanRank')
            plt.xlabel('Epochs')
            plt.savefig(outputDirTail + '/meanrank_tail-' + str(k) + '.pdf')
            plt.close()

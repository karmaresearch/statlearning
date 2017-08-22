import os
import json
import datetime
import logging
from collections import namedtuple

class Logger:
    def _write(self):
        # Write the JSON to file
        with open(self.pathFile, 'w') as outfile:
            json.dump(self.json, outfile)

    def __init__(self, logdirectory, logfilename, loglevel='debug'):
        # Create a new file
        idx = 0
        for file in os.listdir(logdirectory):
           if not file.startswith('.'):
            file, ext = os.path.splitext(logdirectory + "/" + file)
            file = os.path.basename(file)
            if file == logfilename and ext[1:].isnumeric():
                next = int(ext[1:])
                if next >= idx:
                    idx = next + 1
        self.pathFile = logdirectory + "/" + logfilename + "." + str(idx)
        self.json = {}
        #logging.basicConfig(filename=self.pathFile + ".log", level=logging.DEBUG)

        self.log = logging.getLogger()
        self.log.setLevel(loglevel)
        # create console handler and set level to info
        handler = logging.StreamHandler()
        handler.setLevel(loglevel)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.log.addHandler(handler)
        # create error file handler and set level to error
        handler = logging.FileHandler(self.pathFile + ".log", "w", encoding=None, delay="true")
        handler.setLevel(loglevel)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.log.addHandler(handler)

    def getLog(self):
        return self.log

    def startExperiment(self, cmdline):
        self.json["start_experiment"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.json["cmdline"] = cmdline
        self.json['epochs'] = []
        self.json['preds'] = {}
        self.json['predsNames'] = {}

    def setInput(self, input):
        # Store the input
        self.json["input_trn_size"] = len(input['train_subs'])
        self.json["input_inferred"] = input["inferred"]

    def addGeneralEpochResults(self, typeTest, epoch, mrr, hits, meanRank):
        Epoch = namedtuple('Epoch', 'typeTest, epoch mrr hits meanRank')
        epoch = Epoch(typeTest, epoch, mrr, hits, meanRank)
        self.json['epochs'].append(epoch)
        self._write()

    def addPredicatesEpochResults(self, typeTest, head, epoch, pred, predName, mrr, hits, meanRank, ntests):
        PredEpoch = namedtuple('PredEpoch', 'typeTest epoch headTail mrr hits meanRank ntests')
        predEpoch = PredEpoch(typeTest, epoch, head, mrr, hits, meanRank, ntests)
        if pred not in self.json['preds']:
            self.json['preds'][pred] = []
        self.json['preds'][pred].append(predEpoch)
        if pred not in self.json['predsNames']:
            self.json['predsNames'][pred] = predName

    def stopExperiment(self):
        self.json["stop_experiment"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self._write()

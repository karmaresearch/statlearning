from launcher import OntoTranse
from base import Experiment
from logger import Logger
from extvars import ParamsPostProcessing

import random
import sys
import os
import trident
import argparse
import pickle

class TestModel(Experiment):

    def __init__(self):
        super(TestModel, self).__init__()
        self.parser2 = argparse.ArgumentParser(prog='Knowledge Graph experiment', conflict_handler='resolve')
        self.parser2.add_argument('--input', type=str, help='', required=True)
        self.parser2.add_argument('--testValid', type=str, help='', default='valid')
        self.parser2.add_argument('--fileTestValid', type=str, help='', required=True)
        self.parser2.add_argument('--vectors', type=str, help='', required=True)
        self.parser2.add_argument('--exvars', type=bool, help='', default=False)
        self.parser2.add_argument('--testmode', type=bool, help='', default=False)
        self.parser2.add_argument('--resultsdir', type=str, help='Path to the directory that contains the results of the experiments', default='.')
        self.parser2.add_argument('--resultsfilename', type=str, help='Name of the file that contains the results of the experiments', default='results')

        self.parser2.add_argument('--post_threshold', type=int, help='', default=0)
        self.parser2.add_argument('--post_nvars', type=int, help='', default=10)
        self.parser2.add_argument('--post_minsize', type=int, help='', default=3)
        self.parser2.add_argument('--post_minmatch', type=float, help='', default=0.5)
        self.parser2.add_argument('--post_defaultincr', type=int, help='', default=10)
        self.parser2.add_argument('--post2_enable', type=bool, help='', default=False)
        self.parser2.add_argument('--post2_increment', type=int, help='', default=10)
        self.parser2.add_argument('--post2_topk', type=int, help='', default=10)

        self.evaluator = OntoTranse

    def run(self):
        self.args = self.parser2.parse_args()

        self.logger = Logger(self.args.resultsdir, self.args.resultsfilename, 'INFO')
        self.logger.startExperiment(sys.argv)
        self.log = self.logger.getLog()

        dbpath = os.path.dirname(self.args.input) + "/vlog"
        self.db = trident.Db(dbpath)

        #set up params postprocessing
        paramspostp = ParamsPostProcessing(self.args.post_threshold,
                                           self.args.post_nvars,
                                           self.args.post_minsize,
                                           self.args.post_minmatch,
                                           self.args.post_defaultincr,
                                           self.args.post2_enable,
                                           self.args.post2_increment,
                                           self.args.post2_topk)

        # Load the vectors
        with open(self.args.vectors, 'rb') as finv:
            vectors = pickle.load(finv)
        # Load the input
        with open(self.args.input, 'rb') as fin:
            data = pickle.load(fin)
        self.logger.setInput(data)
        self.dictr = data['relations']
        self.dicte = data['entities']

        nameSetToTest = self.args.testValid
        ev = vectors['ev']
        N = len(data['entities'])
        true_triples = data['train_subs'] + data['test_subs'] + data['valid_subs']
        with open(self.args.fileTestValid, 'rb') as fileTV:
            setToTest = pickle.load(fileTV)

        testMode = self.args.testmode

        while True:
            tester = self.evaluator(self.log, setToTest,
                                    true_triples,
                                    N, ev.getNExtVars(),
                                    ev, data['relations'],
                                    data['entities'],
                                    paramspostp,
                                    self.neval)

            # Do the test
            pos_v, fpos_v = tester.positions(vectors['model'], self.args.exvars, ev, self.db)
            fmrr_valid = self.ranking_scores(pos_v, fpos_v, 0, nameSetToTest)
            if testMode:
                print(fmrr_valid)
                # Change the parameters
            else:
                break


        self.logger.stopExperiment()

if __name__ == '__main__':
    TestModel().run()


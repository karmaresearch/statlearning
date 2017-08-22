#!/usr/bin/env python

import numpy as np
from base import Experiment, FilteredRankingEval
from skge import TransE, PairwiseStochasticTrainer

import os
import trident
import taxonomy
import ontology
from logger import Logger
import sys

class OntoTranse(FilteredRankingEval):

    def prepare(self, mdl, p):
        self.ER = mdl.E + mdl.R[p]

    def scores_o(self, mdl, s, p):
        return -np.sum(np.abs(self.ER[s] - mdl.E), axis=1)

    def scores_s(self, mdl, o, p):
        return -np.sum(np.abs(self.ER - mdl.E[o]), axis=1)


class Launcher(Experiment):

    def __init__(self):
        super(Launcher, self).__init__()
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components')
        #self.parser.add_argument('--taxonomy', type=str, default='', help='Path to file that contains the taxonomy of classes')
        self.parser.add_argument('--resultsdir', type=str, help='Path to the directory that contains the results of the experiments', default='.')
        self.parser.add_argument('--resultsfilename', type=str, help='Name of the file that contains the results of the experiments', default='results')
        self.parser.add_argument('--extvars', type=bool, default=False, help='Load and use the existential variables')
        self.parser.add_argument('--onlytraining', type=bool, default=False, help='If set to true then no validation is performed')
        self.parser.add_argument('--thre_min', type=int, default='10', help='Minimum number of elements to consider a subgraph')
        self.parser.add_argument('--valid_sample', type=float, default='1', help='Validate the best model only using a sample of the validate dataset')
        self.parser.add_argument('--loglevel', type=str, default='INFO', help='Log level')
        self.evaluator = OntoTranse

    def setup_trainer(self, sz, sampler, ev, existing_model=None):
        model = TransE(sz, self.args.ncomp, inite=self.args.inite, initr=self.args.initr, ev=ev, model=existing_model)
        trainer = PairwiseStochasticTrainer(
            model,
            nbatches=self.args.nb,
            margin=self.args.margin,
            max_epochs=self.args.me,
            learning_rate=self.args.lr,
            samplef=sampler.sample,
            post_epoch=[self.callback]
        )
        return trainer

    def run(self):
        # parse comandline arguments
        self.args = self.parser.parse_args()
        if self.args.mode != 'rank':
            raise ValueError('Unknown experiment mode (%s)' % self.args.mode)
        self.callback = self.ranking_callback

        self.logger = Logger(self.args.resultsdir, self.args.resultsfilename, self.args.loglevel)
        self.logger.startExperiment(sys.argv)
        self.log = self.logger.getLog()

        self.ev = None
        if self.args.extvars:
            self.evActive = True
        self.existing_model = None

        dbpath = os.path.dirname(self.args.fin) + "/vlog"
        self.db = trident.Db(dbpath)
        self.minSize = self.args.thre_min
        #self.maxSize = self.args.thre_max

        self.train()
        self.logger.stopExperiment()

if __name__ == '__main__':
    Launcher().run()

import numpy as np
from skge.base import Model
from skge.util import grad_sum_matrix, unzip_triples, ccorr, cconv
from skge.param import normless1
import skge.actfun as af


class HolE(Model):

    def __init__(self, *args, **kwargs):
        super(HolE, self).__init__(*args, **kwargs)
        self.add_hyperparam('sz', args[0])
        self.add_hyperparam('ncomp', args[1])
        self.add_hyperparam('rparam', kwargs.pop('rparam', 0.0))
        self.add_hyperparam('af', kwargs.pop('af', af.Sigmoid))
        
        newsize = self.sz[0]
        # Add new embeddings for the existential variables
        ev = kwargs['ev']
        newsize += ev.getNExtVars()
        self.add_param('E', (newsize, self.ncomp), post=normless1)
        self.add_param('R', (self.sz[2], self.ncomp))

        # Copy the embeddings if already existing
        self.updateOffsetE = 0 # This parameter is used to force the update of the gradients only to the subgraphs
        if 'model' in kwargs:
            if kwargs['model'] is not None:
                self.origmodel = kwargs['model']
                hole_model = self.origmodel['model']
                self.E[:len(hole_model.E)] = hole_model.E
                self.R[:len(hole_model.R)] = hole_model.R
                self.updateOffsetE = self.sz[0]
        

    def _scores(self, ss, ps, os):
        return np.sum(self.R[ps] * ccorr(self.E[ss], self.E[os]), axis=1)

    def _gradients(self, xys):
        ss, ps, os, ys = unzip_triples(xys, with_ys=True)

        yscores = ys * self._scores(ss, ps, os)
        self.loss = np.sum(np.logaddexp(0, -yscores))
        #preds = af.Sigmoid.f(yscores)
        fs = -(ys * af.Sigmoid.f(-yscores))[:, np.newaxis]
        #self.loss -= np.sum(np.log(preds))

        ridx, Sm, n = grad_sum_matrix(ps)
        gr = Sm.dot(fs * ccorr(self.E[ss], self.E[os])) / n
        gr += self.rparam * self.R[ridx]

        eidx, Sm, n = grad_sum_matrix(list(ss) + list(os))
        ge = Sm.dot(np.vstack((
            fs * ccorr(self.R[ps], self.E[os]),
            fs * cconv(self.E[ss], self.R[ps])
        ))) / n
        ge += self.rparam * self.E[eidx]

        return {'E': (ge, eidx), 'R':(gr, ridx)}

    def _pairwise_gradients(self, pxs, nxs):
        # indices of positive examples
        sp, pp, op = unzip_triples(pxs)
        # indices of negative examples
        sn, pn, on = unzip_triples(nxs)

        pscores = self.af.f(self._scores(sp, pp, op))
        nscores = self.af.f(self._scores(sn, pn, on))

        #print("avg = %f/%f, min = %f/%f, max = %f/%f" % (pscores.mean(), nscores.mean(), pscores.min(), nscores.min(), pscores.max(), nscores.max()))

        # find examples that violate margin
        ind = np.where(nscores + self.margin > pscores)[0]
        self.nviolations = len(ind)
        if len(ind) == 0:
            return

        # aux vars
        sp, sn = list(sp[ind]), list(sn[ind])
        op, on = list(op[ind]), list(on[ind])
        pp, pn = list(pp[ind]), list(pn[ind])
        gpscores = -self.af.g_given_f(pscores[ind])[:, np.newaxis]
        gnscores = self.af.g_given_f(nscores[ind])[:, np.newaxis]

        # object role gradients
        ridx, Sm, n = grad_sum_matrix(pp + pn)
        grp = gpscores * ccorr(self.E[sp], self.E[op])
        grn = gnscores * ccorr(self.E[sn], self.E[on])
        #gr = (Sm.dot(np.vstack((grp, grn))) + self.rparam * self.R[ridx]) / n
        gr = Sm.dot(np.vstack((grp, grn))) / n
        gr += self.rparam * self.R[ridx]

        # filler gradients
        eidx, Sm, n = grad_sum_matrix(sp + sn + op + on)
        geip = gpscores * ccorr(self.R[pp], self.E[op])
        gein = gnscores * ccorr(self.R[pn], self.E[on])
        gejp = gpscores * cconv(self.E[sp], self.R[pp])
        gejn = gnscores * cconv(self.E[sn], self.R[pn])
        ge = Sm.dot(np.vstack((geip, gein, gejp, gejn))) / n
        #ge += self.rparam * self.E[eidx]
        
        if self.updateOffsetE > 0:
            cond = np.where(eidx >= self.updateOffsetE)
            if len(cond[0]) > 0:
                ge = ge[cond[0][0]:]
                eidx = eidx[cond[0][0]:]
            else:
                ge = []
                eidx = []
            gr = []
            ridx = []

        return {'E': (ge, eidx), 'R':(gr, ridx)}


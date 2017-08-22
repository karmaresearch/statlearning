#!/bin/sh

BDIR=./statlearning
PT=${BDIR}/hole/kg
PT=${PT}:${BDIR}/ontotranse
PT=${PT}:${BDIR}/scikit-kge
PT=${PT}:/usr/local/trident
export PYTHONPATH=$PT

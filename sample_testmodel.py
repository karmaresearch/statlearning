import sys
import pickle
import random

input = sys.argv[1]
nameset = sys.argv[2]
output = sys.argv[3]
sample = float(sys.argv[4])

with open(input, 'rb') as fin:
    data = pickle.load(fin)
origSetToTest = data[nameset]
setToTest = []
if sample < 1:
    for t in origSetToTest:
        # sample the test data
        coin = random.random()
        if coin < sample:
            setToTest.append(t)
else:
    setToTest = origSetToTest

with open(output, 'wb') as fout:
    pickle.dump(setToTest, fout)

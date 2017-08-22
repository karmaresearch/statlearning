import sys
import json
from operator import itemgetter

side1 = sys.argv[1]
side2 = sys.argv[2]

with open(side1, 'rt') as ts1:
    res1 = json.load(ts1)

with open(side2, 'rt') as ts2:
    res2 = json.load(ts2)

# Compare the two:
increasesH = []
decreasesH = []
unchangedH = []
increasesT = []
decreasesT = []
unchangedT = []
for pred1, vpred1 in res1['preds'].items():
    # Get the results for the other side
    vpred2 = res2['preds'][pred1]
    head1 = vpred1[0]
    tail1 = vpred1[1]

    head2 = vpred2[0]
    tail2 = vpred2[1]

    value1 = head1[5] # mean position
    value2 = head2[5] # mean position
    if value1 != value2:
        if value1 < value2:
            increasesH.append((pred1, value1, value2, value2 / value1, head1[6]))
        else:
            decreasesH.append((pred1, value1, value2, value2 / value1, head1[6]))
    else:
        unchangedH.append((pred1, head1[6]))
    value1 = tail1[5] # mean position
    value2 = tail2[5] # mean position
    if value1 != value2:
        if value1 < value2:
            increasesT.append((pred1, value1, value2, value2 / value1, tail1[6]))
        else:
            decreasesT.append((pred1, value1, value2, value2 / value1, tail1[6]))
    else:
        unchangedT.append((pred1, tail1[6]))

print("Number unchanged predicates: %d " % len(unchangedH))
# Sort them
increasesH_new = []
for inc in increasesH:
    increasesH_new.append((inc[0], inc[1], inc[2], inc[3], inc[4], inc[4] * inc[2] - inc[4] * inc[1]))
increasesH = increasesH_new
increases = sorted(increasesH, key=itemgetter(5))[::-1]
print("Increased %d Predicates (head) -- BAD" % len(increases))
for inc in increases:
    print("    %0.2f => %0.2f ntests=%d id=%s %s diff=%f" % (inc[1], inc[2], inc[4], inc[0], res1['predsNames'][inc[0]], inc[5]))

sumDecrease = 0
sumTotal = 0
decreasesH_new = []
for inc in decreasesH:
    decreasesH_new.append((inc[0], inc[1], inc[2], inc[3], inc[4], inc[4] * inc[1] - inc[4] * inc[2]))
decreasesH = decreasesH_new
decreases = sorted(decreasesH, key=itemgetter(5))[::-1]
print("Decreased %d Predicates (head) -- GOOD" % len(decreases))
for inc in decreases:
    print("    %0.2f => %0.2f ntests=%d id=%s %s diff=%f" % (inc[1], inc[2], inc[4], inc[0], res1['predsNames'][inc[0]], inc[5]))
    sumDecrease += inc[5]
    sumTotal += inc[1] * inc[4]
print("Sum difference %d total %d" % (sumDecrease, sumTotal))

#print("Unchanged Predicates (head) -- NEUTRAL")
#for inc in unchangedH:
#     print("    ntests=%d id=%s %s" % (inc[1], inc[0], res1['predsNames'][inc[0]]))


increases = sorted(increasesT, key=itemgetter(3))[::-1]
print("Increased Predicates (tail) -- BAD")
for inc in increases:
    print("    %0.2f => %0.2f ntests=%d id=%s %s" % (inc[1], inc[2], inc[4], inc[0], res1['predsNames'][inc[0]]))
print("Decreased Predicates (tail) -- GOOD")
for inc in decreasesT:
    print("    %0.2f => %0.2f ntests=%d id=%s %s" % (inc[1], inc[2], inc[4], inc[0], res1['predsNames'][inc[0]]))
#print("Unchanged Predicates (tail) -- NEUTRAL")
#for inc in unchangedT:
#    print("    ntests=%d id=%s %s" % (inc[1], inc[0], res1['predsNames'][inc[0]]))

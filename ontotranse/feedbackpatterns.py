class FeedbackPattern:

    def __init__(self):
        self.associations = {}
        self.feebacksMap = {}

    def getAssociatedPatterns(self, po):
        if po in self.associations:
            return self.associations[po]
        else:
            return []

    def likelyTrue(self, s, po):
        if po in self.feebacksMap and s in self.feebacksMap[po]:
            return True
        else:
            return False

    def addToLikelyTrue(self, alls, po):
        if po not in self.feebacksMap:
            self.feebacksMap[po] = set()
        for s in alls:
            self.feebacksMap[po].add(s)
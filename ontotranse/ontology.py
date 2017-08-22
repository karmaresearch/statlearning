import pickle
from collections import namedtuple

class Ontology:

    def __init__(self):
        self._isLoaded = False

    def _load_extvars(self, onto, dictr, dicte, ontop):
        vars = {}
        idSomeValuesFrom = None
        if "<http://www.w3.org/2002/07/owl#someValuesFrom>" in ontop:
            idSomeValuesFrom = ontop["<http://www.w3.org/2002/07/owl#someValuesFrom>"]
        idOnProp = None
        if "<http://www.w3.org/2002/07/owl#onProperty>" in ontop:
            idOnProp = ontop["<http://www.w3.org/2002/07/owl#onProperty>"]

        if idSomeValuesFrom and idOnProp and idSomeValuesFrom in onto and idOnProp in onto:
            for triple in onto[idSomeValuesFrom]:
                vars[triple[0]] = {'sv' : triple[1]}
            for k in vars.keys():
                lis = onto[idOnProp]
                for t in lis:
                    if t[0] == k:
                        vars[k]['onp'] = t[1]

        DetailExtVar = namedtuple('DetailExtVar', 'id sv prop textkey textp textc key')
        exvarsPerKey = {}
        idKey = 0
        for k, v in vars.items():
            if len(v) == 2:
                p = v['onp']
                textp = dicte[p]
                # Search in dictr the ID for the 'p'
                idxp = 0
                found = False
                while idxp < len(dictr):
                    if dictr[idxp] == textp:
                        found = True
                        break
                    idxp += 1
                if found:
                    p = idxp
                    c = v['sv']
                    if k not in exvarsPerKey:
                       exvarsPerKey[k] = []
                    exvarsPerKey[k].append(DetailExtVar(id=idKey, key=k, sv=c, prop=p, textkey=dicte[k], textp=textp, textc=dicte[c]))
                    idKey += 1
                else:
                    print("Predicate not found: %s" % textp)

        # Store the variables in the ontology global object
        self._exvarsPerKey = exvarsPerKey

        # Create a data structure where each predicate is associated to all existential variables
        exvarsPerPred = {}
        for k, v in exvarsPerKey.items():
            for el in v:
                p = el.prop
                if p not in exvarsPerPred:
                    exvarsPerPred[p] = []
                exvarsPerPred[p].append(el)
        self._exvarsPerPred = exvarsPerPred

    def _load_types(self, subs, typepred):
        instances = {}
        for t in subs:
            if t[2] == typepred:
               if t[0] not in instances:
                   instances[t[0]] = []
               instances[t[0]].append(t[1])
        self._types = instances

    def load(self, ontologyFile):
        data = pickle.load(open(ontologyFile, 'rb'))

        # onto contains the ontological triples grouped by predicate
        onto = data["onto"]
        # I need these three data structures to translate the properties using the relation IDs
        dictr = data["relations"]
        dicte = data["entities"]
        ontop = data["ontopreds"]
        # Load all existential variables
        self._load_extvars(onto, dictr, dicte, ontop)

        # Get the ID for the type predicate
        typeID = -1
        for k,v in dictr.items():
            if v == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
                typeID = k
                break
        self._typeID = typeID
        # Load all types associated with the entities
        self._load_types(data["train_subs"], typeID)
        self._isLoaded = True

    def getTypeID(self):
        return self._typeID

    def isLoaded(self):
        return self._isLoaded

    def getExVarsPerKey(self):
        return self._exvarsPerKey

    def getExVarsCoupledWithP(self, p):
        if p in self._exvarsPerPred:
            return self._exvarsPerPred[p]
        else:
            return None

    def isInstanceOfType(self, instance, clazz):
        if instance in self._types:
            for c in self._types[instance]:
                if c == clazz:
                    return True
            return False
        else:
            return False

_ontology = Ontology()

def getOntology():
    return _ontology

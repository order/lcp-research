import cPickle as pickle

def dump(obj,filename):
    FH = open(filename,'wb')
    pickle.dump(obj,FH)
    FH.close()

def load(filename):
    FH = open(filename,'rb')
    data = pickle.load(FH)
    FH.close()
    return data

import cPickle as pickle
import time

def dump(obj,filename):
    FH = open(filename,'wb')
    pickle.dump(obj,FH)
    FH.close()

def load(filename):
    FH = open(filename,'rb')
    data = pickle.load(FH)
    FH.close()
    return data

def do_or_load(do_fn, filename, do,phase=None):
    """
    If "do" is true, run the do_fn and save result
    Else: just load the file
    """
    start = time.time()
    if not do:
        if phase:
            print 'Skipping',phase
        obj = load(filename)
    else:
        if phase:
            print 'Running',phase
        obj = do_fn()
        dump(obj,filename)
    if phase:
        print '\tDone. ({0}s)'.format(time.time() - start)
    return obj  

class KwargParser(object):
    """
    Simple parser 
    """
    
    def __init__(self):
        self.defaults = {}
        self.types = {}
        self.mandatory = set()
        
    def add(self,key,*default):

        assert(2 >= len(default))
        
        if 0 == len(default):
            self.mandatory.add(key)
               
        if 1 <= len(default):
            self.defaults[key] = default[0]
            
        if 2 == len(default):
            if type(default[1]) is list:
                self.types[key] = default[1]
            else:
                self.types[key] = [default[1]]

            
    def parse(self,D):
        # Check if there are unexpected options
        keys = set(self.defaults.keys()) | self.mandatory
        weird = set(D.keys()) - keys
        if len(weird) > 0:
            print 'Unknown options:'
            for k in weird:
                print '\t{0}: {1}'.format(k,D[k])
            assert(0 == len(weird))
            
        missing = self.mandatory - set(D.keys())
        if len(missing) > 0:
            print 'Missing options:', ','.join(map(str,missing))
            assert(0 == len(missing))

        ret = dict(self.defaults)
        ret.update(D)

        for k in self.types:
            if not type(ret[k]) in self.types[k]:
                print '{0}:{1} not in set of correct types {2}'\
                    .format(k,ret[k],self.types[k])

        return ret

            
        
        
        

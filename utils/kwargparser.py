class KwargParser(object):
    """
    Simple parser for dicts
    """
    
    def __init__(self):
        self.defaults = {}
        self.types = {}
        self.mandatory = set()
        self.optional = set()
        
    def add(self,key,*default):
        """
        Any keys added this way are always have an 
        assigned value after parsing
        """
        assert(2 >= len(default))
        
        if 0 == len(default):
            # All keys added this way are mandatory
            self.mandatory.add(key)
               
        if 1 <= len(default):
            # Key and default: declares a default value
            self.defaults[key] = default[0]
            
        if 2 == len(default):
            # Key, default, and list of acceptable types
            if type(default[1]) is list:
                self.types[key] = default[1]
            else:
                assert(isinstance(int,type))
                self.types[key] = [default[1]]
                
    def add_optional(self,key,default=None):
        """
        Keys that may or may not be assigned.
        """
        self.optional.add(key)

        if default:
            if type(default) is list:
                self.types[key] = default
            else:
                assert(isinstance(int,type))
                self.types[key] = [default]
            
    def parse(self,D):
        # Check if there are unexpected options
        keys = set(self.defaults.keys()) | self.mandatory | self.optional
        weird = set(D.keys()) - keys
        if len(weird) > 0:
            print 'Unknown options:'
            for k in weird:
                print '\t{0}: {1}'.format(k,D[k])
            assert(0 == len(weird))
            
        # Check if there are missing options
        keys -= self.optional # remove optional keys
        missing = self.mandatory - set(D.keys())
        if len(missing) > 0:
            print 'Missing options:', ','.join(map(str,missing))
            assert(0 == len(missing))

        ret = dict(self.defaults)
        ret.update(D)

        for k in self.types:
            if k not in ret:
                assert(k in self.optional)
                continue
            if not type(ret[k]) in self.types[k]:
                print '{0}:{1} not in set of correct types {2} ({3})'\
                    .format(k,ret[k],self.types[k],type(ret[k]))
                assert(type(ret[k]) in self.types[k])

        return ret

            
        
        
        

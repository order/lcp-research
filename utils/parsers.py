import types
import warnings
import importlib
import os

class KwargParser(object):
    """
    Simple parser for dicts
    """
    
    def __init__(self):
        self.defaults = {}
        self.types = {}
        self.mandatory = set()
        self.optional = set()
        self.permissive = False
        
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
            for k in weird:
                warnings.warn('Unknown option: {0}'.format(k))
        if not self.permissive:
            assert(0 == len(weird))
            
        # Check if there are missing options
        keys -= self.optional # remove optional keys
        missing = self.mandatory - set(D.keys())
        if len(missing) > 0:
            raise Exception('Missing options: '+','.join(map(str,missing)))

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

class ConfigParser(object):
    """
    Read in a file-based configuration

    Enteries should be like:
    foo = baz

    A "handler" is a function that transforms
    the value. Default is using "eval"; but the
    default can be changed and special cases can be used
    (like identity, or load_class)
    """
    comment_symbol = "#"
    def __init__(self,filename):
        self.fh = open(filename,'r')
        self.handlers = {}
        self.prefix_handlers = {}
        self.default_handler = eval

    def add_handler(self,key,fn):
        assert(isinstance(fn,types.FunctionType))
        self.handlers[key] = fn

    def add_prefix_handler(self,prefix,fn):
        assert(isinstance(fn,types.FunctionType))
        self.prefix_handlers[prefix] = fn

    def get_handler(self,S):
        if S in self.handlers:
            return self.handlers[S]
        
        for (prefix,handler) in self.prefix_handlers.items():
            if S.startswith(prefix):
                return handler
        return None            
        
    def parse(self):
        args = {}
        for line in self.fh:
            # Remove everything after the comment symbol
            if self.comment_symbol in line:
                index = line.find(self.comment_symbol)
                line = line[:index]
            line = line.strip() # strip

            # Don't do anything if empty
            if 0 == len(line):
                continue

            # Otherwise of form "foo = baz"
            print "Parsing",line
            assert('=' in line)
            (key,val_str) = [x.strip() for x in line.split('=')]
            handler = self.get_handler(key)
            if handler:
                val = handler(val_str)
            else:
                # eval is default handler
                try:
                    val = self.default_handler(val_str)
                except:
                    warnings.warn('{0} not handled; leaving as string'\
                                  .format(val_str))
                    val = val_str
            args[key] = val
        return args


def hier_key_dict(D,sep):
    """
    Takes a dictionary with keys like "key1.key2.key3 = val" and
    breaks into a nested dictionary like:
    key1 = {key2 = {key3 = val}}
    """
    NewD = {}
    for key in D:
        split_key = [x.strip() for x in key.split(sep)]
        CurrD = NewD
        for subkey in split_key[:-1]:
            if subkey not in CurrD:
                CurrD[subkey] = {}
            CurrD = CurrD[subkey]
        CurrD[split_key[-1]] = D[key]

    return NewD
            

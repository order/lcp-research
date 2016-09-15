import numpy as np
import scipy.sparse as sps
import tarfile

from utils import is_int
import StringIO

class Archiver(object):
    def __init__(self,**kwargs):
        self.data = {}
        self.add(**kwargs)

    def add(self,**kwargs):
        for v in kwargs.values():
            assert(isinstance(v,np.ndarray)
                   or isinstance(v, sps.spmatrix))
        self.data.update(kwargs)


    def clear(self):
        self.data = {}

    def write(self,filename):
        tar = tarfile.open(filename,"w:gz")
        for (k,v) in self.data.items():
            (arma_type,packed) = pack(v)
            buff = np.getbuffer(packed)

            var_file_name = k + '.' + arma_type
            info = tarfile.TarInfo(name=var_file_name)
            info.size=(len(buff))
            tar.addfile(tarinfo=info,fileobj=StringIO.StringIO(buff))
        tar.close()

class Unarchiver(object):
    def __init__(self,filename):
        self.data = {}
        
        tar = tarfile.open(filename, "r:gz")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            content=f.read()
            (var_name,arma_type) = member.name.split('.')

            # Check the reported type
            dtype = np.float
            if(arma_type[0] == 'i'):
                dtype=np.integer
            elif(arma_type[0] == 'u'):
                dtype=np.unsignedinteger
            raw_var = np.frombuffer(content,dtype=dtype)
            
            self.data[var_name] = eval('unpack_{0}(raw_var)'.format(arma_type))
        self.__dict__.update(self.data)
        tar.close()
        
        
def pack(A):
    # Currently only working for vecs and mats
    assert isinstance(A,np.ndarray)
    assert A.dtype.type == np.double

    if 1 == A.ndim:
        return ('vec',np.hstack([A.shape,A]))
    assert 2 == A.ndim
    # ARMA is Fortran/column order
    return ('mat',np.hstack([A.shape,A.flatten(order='F')]))
    

def unpack_vec(vec):
    N = vec[0]
    assert N % 1 < 1e-15
    N = int(N)
    assert (N+1) == vec.size
    
    return vec[1:];

def unpack_uvec(vec):
    return unpack_vec(vec).astype(np.integer)

def unpack_mat(vec):
    N = vec[0]
    D = vec[1]
    assert N % 1 < 1e-15
    assert D % 1 < 1e-15
    N = int(N)
    D = int(D)
    assert (N*D + 2) == vec.size

    return np.reshape(vec[2:],(N,D),order='F')

def unpack_sp_mat(A):
    assert A.size >= 3
    assert is_int(A[:3])
    (R,C,nnz) = map(int,A[:3])
    assert((3 + 3*nnz) == A.size)

    # Armadillo works in Fortran format
    triples = np.reshape(A[(-3*nnz):],(3,nnz),order='F')

    rows = triples[0,:]
    cols = triples[1,:]
    data = triples[2,:]
    assert(is_int(rows))
    assert(is_int(cols))
    
    rows = rows.astype(np.integer)
    cols = cols.astype(np.integer)    
    return sps.coo_matrix((data,(rows,cols)),shape=(R,C))


def prep_lines(lines):
    lines = [line.split('#',1)[0].strip() for line in lines]
    lines = [line for line in lines if line]
    return list(reversed(lines))

def read_medit_mesh(filename):
    # Read in file in INRIA's .medit format
    FH = open(filename,"r")
    lines = FH.readlines()
    lines = prep_lines(lines)

    names = ['vertices','edges','triangles','tetrahedra']
    for name in names:
        exec(name + ' = []')
    while lines:
        line = lines.pop()
        while lines and line.lower() not in names:
            line = lines.pop()
        if not lines:
            break
        name = line.lower()
        assert name in names
        
        n = int(lines.pop())
        objs = []
        for _ in xrange(n):
            line = lines.pop()
            tokens = line.split()
            obj = map(float,tokens[:-1]) # Ignore boundary marker information
            objs.append(obj)
        objs = np.array(objs) 
        exec(name + ' = objs')

    return vertices,edges,triangles,tetrahedra

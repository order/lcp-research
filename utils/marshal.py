import numpy as np

class Marshaller(object):
    def __init__(self):
        self.objects = []
        self.scalars = tuple([int,float,np.int64])
        self.types = tuple(list(self.scalars) + [np.ndarray])

    def add(self,obj):
        # Is a number-y type
        if(isinstance(obj,list)):
            obj = np.array(obj)
        assert(type(obj) in self.types)
        if isinstance(obj,np.ndarray):
            # Is a vector,array, or cube
            assert(1<= len(obj.shape) <= 3)
        self.objects.append(obj)

    def extend(self,objs):
        [self.add(obj) for obj in objs]

    def save(self,filename):
        num_objs = len(self.objects)
        data_size = 0
        
        desc = []
        data = []
        for obj in self.objects:
            if isinstance(obj,self.scalars):
                desc.append(0)
                data.append(obj)
                data_size += 1
            else:
                assert(isinstance(obj,np.ndarray))
                desc.append(len(obj.shape))
                desc.append(obj.shape)
                
                data.append(obj.flatten(order='F'))
                data_size += np.prod(obj.shape)

        desc = np.hstack(desc)
        header_size = 3 + desc.size
        output = np.hstack([num_objs,header_size,data_size]
                           + [desc]
                           + data).astype(np.double)
        assert((header_size + data_size,) == output.shape)
        output.tofile(filename)

    def load(self,filename):
        A = np.fromfile(filename,dtype=np.double)
        (N,) = A.shape
        
        num_objs =   int(A[0])
        header_size = int(A[1])
        data_size =  int(A[2])
        assert(data_size + header_size == N)

        objs = []
        h = 3
        d = header_size
        for i in xrange(num_objs):
            obj_dim = int(A[h])
            h += 1
            if 0 == obj_dim:
                objs.append(A[d])
                d += 1
            else:
                shape = A[h:(h+obj_dim)].astype(np.int)
                linlen = np.prod(shape)
                h += obj_dim

                obj = np.reshape(A[d:(d+linlen)],shape,order='F')
                objs.append(obj)
                d += linlen
        return objs

if __name__ == '__main__':
    marshaller = Marshaller()
    marshaller.add(np.arange(5))
    marshaller.add(2)
    marshaller.add(np.ones((4,3)))

    marshaller.save('test.bin')
    objs = marshaller.load('test.bin')
    print objs

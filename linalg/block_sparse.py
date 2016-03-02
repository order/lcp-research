class bsparse_vect(object):
    def __init__(self,block_sizes):
        self.block_sizes = block_sizes
        self.b_size = len(block_sizes)
        self.b_shape = tuple(self.b_size)
        self.blocks = [None]*self.b_size

        self.size = np.sum(block_sizes)
        self.shape = tuple(self.size)

    def assign(self,x,block_id):
        bl_sz = self.block_sizes[block_id]
        assert((bl_sz,) == x.shape)
        

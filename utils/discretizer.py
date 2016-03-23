class IrregularSplit(object):
    def __init__(self,cutpoints):
        (K,D) = cpts.shape
        self.cpts = cutpoints
        self.dim = D
        self.num_cpts = K
        
    def cell(self,d,k):
        """
        Get the kth cell boundary in dimension d
        E.g.:
        |x|x|x|
        There are 4 cut points, and 3 cells
        defined by the flanking cutpoints
        """
        assert(0 <= d < self.dim)
        assert(0 <= k < self.num_cpts-1)
        lo = self.cpts[k,d]
        hi = self.cpts[k+1,d]
        return (lo,hi)

    def boundary(self,d):
        return (self.cpts[0,d],self.cpts[-1,d])

class IrregularDiscretizer(object):
    def __init__(self,splits):
        self.splits = splits
    def discretize(self,points):
        (N,D) = points.shape
        assert(self.splits.dim == D)
        K = self.splits.num_cpts

        Coords = np.empty((N,D))
        for d in xrange(D):
            for k in xrange(K-1):
                # Get the boundary of the kth cell in dimension d
                (lo,hi) = self.splits.cell(d,k)

                # Figure out which elements of samples are in this
                # Cell
                indx = np.where(np.logical_and(points[:,d] >= lo,
                                              points[:,d] < hi))

                # Set the discrete coord for this
                Coords[indx,d] = k

            (left,right) = self.splits.boundary(d)
            oobs = np.where(np.logical_or(points[:,d] < left,
                                          points[:,d] >= right))
            Coords[oobs,d] = np.nan
            


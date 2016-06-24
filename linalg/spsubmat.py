import scipy.sparse as sps

def spsubmat(S,I,J):
    return (S.tocsr()[I,:]).tocsc()[:,J]

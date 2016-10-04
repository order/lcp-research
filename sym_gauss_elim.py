from sympy import *
from sympy.matrices import *

import re

def entry_eliminate(X,Y):
    assert X.shape == Y.shape
    if X == Y:
        return ZeroMatrix(*X.shape)
    return X - Y

def eliminate(G,c,r):
    #eliminate column c using row r
    assert not isinstance(G[r,c],ZeroMatrix)
    (R,C) = G.shape
    G.row_op(r,lambda X,j: G[r,c].I * X)
    for i in xrange(R):
        if i == r:
            continue
        if ZeroMatrix(*G[i,c].shape) == G[i,c]:
            continue
        G.row_op(i,lambda X,j: entry_eliminate(X,G[i,c] * G[r,j]))

def original_gordon_calc():
    N = 3
    K = 2
    Y = MatrixSymbol('Y',N,N)
    X = MatrixSymbol('X',N,N)
    M = MatrixSymbol('M',N,N)
    P = MatrixSymbol('P',N,K)
    g = MatrixSymbol('g',N,1)
    r = MatrixSymbol('r',N,1)
    ZNN = ZeroMatrix(N,N)
    ZNK = ZeroMatrix(N,K)
    z = ZeroMatrix(N,1)
    I = Identity(N)

    print "Full Netwon system:"
    print "Variable block order: [x y w]"
    print
    G = Matrix([[Y,X,ZNK,g], [ZNN-M,I,ZNK,r],[ZNN-I,I,P,z]])
    pprint(G)
    (R,C) = G.shape
    eliminate(G,0,2)
    print "Eliminate first column using last row:"
    pprint(G)
    eliminate(G,1,0)
    print "Eliminate second column using first row:"
    pprint(G)

def mixed_gordon_calc():
    N = 5
    NB = 3
    NF = 2
    K = 2
    NK = 2
    B = MatrixSymbol('B',NB,NB)
    S = MatrixSymbol('S',NB,NB)

    for x in ['f','b']:
        exec("P{0} = MatrixSymbol('\\Phi_{{{0}}}',N{1},K)".format(x,x.upper()))
        exec("I{0} = Identity(N{0})".format(x.upper()))
        exec("r{0} = MatrixSymbol('r_{0}',N{1},1)".format(x,x.upper()))
        exec("z{0} = ZeroMatrix(N{1},1)".format(x,x.upper()))
        for y in ['f','b','k']:
            exec("M{0}{1} = MatrixSymbol('M_{{{0}{1}}}',N{2},N{3})".format(x,y,x.upper(),y.upper()))
            exec("Z{0}{1} = ZeroMatrix(N{0},N{1})".format(x.upper(),y.upper()))
    g = MatrixSymbol('g',NB,1)
    z = ZeroMatrix(N,1)

    G = Matrix([[ZBF, S,   B,   ZBK, g],
                [Mff, Mfb, ZFB, ZFK, rf],
                [Mbf, Mbb,-IB,  ZBK, rb],
                [IF,  ZFB, ZFB,-Pf,  zf],
                [ZBF, IB, -IB, -Pb,  zb]])
    print "Variable block order: [x s w]"
    print
    print '-'*80
    pprint(G)
    
    eliminate(G,0,3)
    eliminate(G,1,4)
    eliminate(G,2,0)
    print '-'*80
    pprint(G)

def matrix_latex(G):
    S = latex(G)
    S = re.sub(r'\\bold\{0\}','0',S)
    S = re.sub(r'\\mathbb\{I\}','I',S)
    S = re.sub(r'\\\\',r'\\\\\n',S)
    return S


print "Original PLCP calculation:"
print '-'*80
#original_gordon_calc()

print
print '='*80
print
print "mPLCP calculation:"
print '-'*80

mixed_gordon_calc()

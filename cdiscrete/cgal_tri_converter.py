import numpy as np
from collections import defaultdict

if __name__ == "__main__":
    
    (vert,tet,adj) = read_cgal_tri("output")
    
    mirror = build_mirror(tet)
    reorient_tets(vert,tet,mirror)
    adj = derive_adj_from_tet(tet,mirror)
    
    write_cgal_tri("output.1",vert,tet,adj)

def read_cgal_tri(filename):
    fh = open(filename,'r')
    lines = [line.strip() for line in fh.readlines()]
    lines = [line for line in lines if line]
    lines = list(reversed(lines)) # reverse for popping
    fh.close()

    verts    = []
    tets = []
    adj = []

    D = int(lines.pop())
    V = int(lines.pop())
    print 'Vertices:'
    verts.append([np.nan]*3) # Explicitly add the oob node
    print 'V[0]:\t',[np.nan]*3, '# OOB node'
    for v in xrange(V):
        coords = map(float,lines.pop().split())
        assert D == len(coords)
        print 'V[{0}]:\t'.format(v),coords
        verts.append(coords)
    print "Tetrahedra:"
    T = int(lines.pop())
    for t in xrange(T):
        idx = map(int,lines.pop().split())
        print 'T[{0}]:\t'.format(t), idx
        assert (D+1) == len(idx)
        tets.append(list(sorted(idx)))
    print "Tet Adjacency:"
    a = 0
    while lines:
        idx = map(int,lines.pop().split())
        print 'A[{0}]:\t'.format(a), idx
        assert (D+1) == len(idx)
        adj.append(list(sorted(idx)))
        a += 1

    return verts, tets, adj

def write_cgal_tri(filename,verts,tets,adj):
    fh = open(filename,'w')
    V = len(verts)
    assert V > 0
    D = len(verts[0])
    T = len(tets)
    
    SS = []
    SS.append(D)
    SS.append(V-1)
    SS.append(verts[1:]) # Don't write OOB node
    SS.append(T)
    SS.append(tets)
    SS.append(adj)

    S = []
    for x in SS:
        if isinstance(x,list):
            for y in x:
                S.append(' '.join(map(str,y)))
        else:
            S.append(str(x))
    fh.write('\n'.join(S))

"""
Get face hashs from a tet
"""
face_hash = lambda x,i: tuple(sorted(x[:i] + x[(i+1):]))
def face_keys(tet):
    assert 4 == len(tet)
    keys = []
    for i in xrange(4):
        key = face_hash(tet,i)
        keys.append(key)
    return keys
    
"""
Turns 4 tetrahedra indices into a 4x3 matrix of
physical coordinates
"""
def tet_point_mat(tet_idx,verts):
    tet_list = [verts[i] for i in tet_idx]
    tet_mat = np.array(tet_list)
    assert (4,3) == tet_mat.shape
    return tet_mat

"""
Every finite face (triangle) belongs to two tetrehedra.
The mirror data structure keeps track of this.
"""
def build_mirror(tets):
    mirror = defaultdict(set)
    for (tet_id,tet) in enumerate(tets):
        assert 4 == len(tet)
        keys = face_keys(tet)
        for key in keys:
            mirror[key].add(tet_id)
            assert len(mirror[key]) <= 2
    return mirror

"""
Takes a tet and a vertex id; 
Returns the mirror tet and the vertex id for the vertex that is different
"""
def get_mirror_tet(mirror,tets,tet_id,vert_id):
    # Get the vertex id list
    tet = list(tets[tet_id])
    assert 4 == len(tet)
    assert vert_id in tet
    
    # Find the index, in the tet list of the supplied vertex id
    idx = tet.index(vert_id)
    
    # Get the face of the tet that doesn't use this vertex
    # This is the 'mirror'
    key = face_hash(tet,idx)

    # Get the pair of tet indices that share that share the mirror
    twins = mirror[key]
    assert tet_id in twins

    # Get the mirror vertex that is not in the face
    mirror_tet_id =  twins - set([tet_id])
    assert 1 == len(mirror_tet_id)
    mirror_tet_id = list(mirror_tet_id)[0] # Get unique tet index
    mirror_tet = tets[mirror_tet_id] # Get the vertex id list

    # What is different between the original vertex id list
    # and the mirror list?
    mirror_vert_id = set(mirror_tet) - set(tet)
    assert 1 == len(mirror_vert_id) 
    mirror_vert_id = list(mirror_vert_id)[0] # Get unique difference

    # Mirror tet vertex id list,
    # In same order as original tet vertex list, but with
    # the indicated vert replaced.
    # Will have the opposite orientation as the original.
    mirror_tet_replace = tet
    mirror_tet_replace[idx] = mirror_vert_id

    # Same vertex ids; perhaps different order
    assert(list(sorted(mirror_tet_replace)) == list(sorted(mirror_tet)))
    
    return mirror_tet_replace

def orient3d(tet_mat):
    assert (4,3) == tet_mat.shape
    assert not np.any(np.isnan(tet_mat))

    # Get difference vectors from the first vertex
    diff = tet_mat[1:,:] - tet_mat[0,:]
    assert (3,3) == diff.shape

    # The sign of the difference matrix is
    # the orientation
    orient = np.linalg.det(diff)
    assert(np.abs(orient) > 1e-6) # Indication of malformed tet
    return np.sign(orient)

def reorient_tets(vertices,tets,mirror):
    for (tet_id,tet) in enumerate(tets):
        if 0 in tet:
            # Mirror world            
            sign = -1
            mirror_tet = get_mirror_tet(mirror,tets,tet_id,0)
            tet_mat = tet_point_mat(mirror_tet,vertices)
        else:
            # Regular world.
            sign = 1
            tet_mat = tet_point_mat(tet,vertices)
            
        assert (4,3) == tet_mat.shape
        assert not np.any(np.isnan(tet_mat))
        if sign*orient3d(tet_mat) < 0:
            # Flip. All even permutations are equivalent
            tets[tet_id][0],tets[tet_id][1] = tets[tet_id][1],tets[tet_id][0]

def derive_adj_from_tet(tets,mirror):
    adj = []
    for (tet_id,tet) in enumerate(tets):
        tet_adj = []
        keys = face_keys(tet)
        for key in keys:
            assert tet_id in mirror[key]
            mirror_tet_id = mirror[key] - set([tet_id])
            assert 1 == len(mirror_tet_id)
            tet_adj.append(list(mirror_tet_id)[0])
        adj.append(tet_adj)
    return adj

def foo(d,sign):
        assert(sign in [-1,1])
        
        return int(2*(d+1) + (sign - 1)/2)

for d in xrange(3):
    for s in [-1,1]:
        print foo(d,s)

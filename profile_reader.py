import pstats
import sys

if 2 == len(sys.argv):
    field = 'tottime'
elif 3 == len(sys.argv):
    field = sys.argv[2]
else:
    print 'Usage: <profile file> <field>'
    quit()

p = pstats.Stats(sys.argv[1])

p.strip_dirs().sort_stats(field).print_stats(15)

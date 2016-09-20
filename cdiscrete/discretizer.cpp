#include "discretizer.h"
BaryCoord::BaryCoord() : oob(true) {}
BaryCoord::BaryCoord(bool aoob,const uvec&aidx,const vec&aw) :
  oob(aoob),indices(aidx),weights(aw) {}

#include "discretizer.h"
BaryCoord::BaryCoord(){}
BaryCoord::BaryCoord(bool aoob,const uvec&aidx,const vec&aw) :
  oob(aoob),indices(aidx),weights(aw) {}

#ifndef __Z_MINOP_INCLUDED__
#define __Z_MINOP_INCLUDED__

#include <armadillo>
#include "misc.h"
#include "tri_mesh.h"
#include "lcp.h"

using namespace std;
using namespace arma;

void generate_minop_mesh(TriMesh & mesh,
                         const string & filename,
                         double edge_length,
                         double min_angle);
void build_minop_lcp(const TriMesh &mesh,
                     const vec & a,
                     LCP & lcp,
                     vec & ans);

double person_rho(const vec &,
                  const vec &);
vec person_rho(const mat &,
               const mat &);

vec jitter_solve(const TriMesh & mesh,
                 const vec & ref_weights,
                 mat & jitter,
                 mat & weight_noise);
#endif

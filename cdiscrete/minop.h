#ifndef __Z_MINOP_INCLUDED__
#define __Z_MINOP_INCLUDED__

#include <armadillo>
#include "misc.h"
#include "tri_mesh.h"
#include "lcp.h"
#include "solver.h"

using namespace std;
using namespace arma;

void generate_minop_mesh(tri_mesh::TriMesh & mesh,
                         const string & filename,
                         double edge_length,
                         double min_angle);
void build_minop_lcp(const tri_mesh::TriMesh &mesh,
                     const vec & a,
                     LCP & lcp,
                     vec & ans);

double pearson_rho(const vec &,
                  const vec &);
vec pearson_rho(const mat &,
               const mat &);

void jitter_solve(const tri_mesh::TriMesh & mesh,
                 const ProjectiveSolver & solver,
                 const PLCP & ref_plcp,
                 const vec & ref_weights,
                 mat & jitter,
                 mat & weight_noise,
                 uint jitter_rounds);
#endif

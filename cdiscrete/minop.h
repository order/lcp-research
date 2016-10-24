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
                         double min_angle,
                         bool write=false);

vec caldera_q(const Points & points,
              const vec & a);

vec bumpy_q(const Points & points,
            const vec & a);

sp_mat build_smoothed_identity(const Points & points,
                             const double bandwidth);

void build_minop_lcp(const tri_mesh::TriMesh &mesh,
                     const vec & a,
                     LCP & lcp,
                     vec & ans);
void build_smoothed_minop_lcp(const tri_mesh::TriMesh &mesh,
                              const vec & a,
                              const double bandwidth,
                              LCP & lcp,
                              vec & ans,
                              sp_mat & I);

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

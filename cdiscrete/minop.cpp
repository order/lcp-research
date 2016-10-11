#include "minop.h"

using namespace tri_mesh;

void generate_minop_mesh(TriMesh & mesh,
                         const string & filename,
                         double edge_length,
                         double angle,
                         bool write){
  cout << "Initial meshing..."<< endl;
  mesh.build_box_boundary({{-1.1,1.1},{-1.1,1.1}});
  mesh.build_circle(zeros<vec>(2),50,1.0);
  mesh.build_circle(zeros<vec>(2),30,1.0/sqrt(2.0));
  mesh.build_circle(zeros<vec>(2),25,0.25);

  cout << "Refining based on (" << angle
       << "," << edge_length <<  ") criterion ..."<< endl;
  mesh.refine(angle,edge_length);
  
  cout << "Optimizing (25 rounds of Lloyd)..."<< endl;
  mesh.lloyd(25);
  mesh.freeze();

  cout << "Number of vertices: " << mesh.number_of_vertices() << endl;
  cout << "Number of faces: " << mesh.number_of_faces() << endl;
 

  // Write initial mesh to file
  if(write){
    cout << "Writing:"
         << "\n\t" << (filename + ".node") << " (Shewchuk node file)"
         << "\n\t" << (filename + ".ele") << " (Shewchuk element file)"
         << "\n\t" << (filename + ".tri") << " (CGAL mesh file)" << endl;
    mesh.write_shewchuk(filename);
    mesh.write_cgal(filename + ".tri");
  }
}

void build_minop_lcp(const TriMesh &mesh,
                     const vec & a,
                     LCP & lcp,
                     vec & ans){
  double off = 1.0; // +ve offset
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  vec sq_dist = sum(pow(points,2),1);

  vec b = sq_dist + off;
  vec c = max(zeros<vec>(N),1 - sq_dist) + off;
  
  ans = arma::max(zeros<vec>(N), arma::min(b,c));
  
  assert(a.n_elem == b.n_elem);
  vec q = join_vert(-a,
                    join_vert(b,c));
  assert(3*N == q.n_elem);
  assert(not all(q >= 0));
 
  vector<sp_mat> E;
  E.push_back(speye(N,N));
  E.push_back(speye(N,N));
  sp_mat M = build_M(E);
  assert(M.is_square());
  assert(3*N == M.n_rows);

  lcp = LCP(M,q);
}

double pearson_rho(const vec & a,
                   const vec & b){
  double mu_a = mean(a);
  double mu_b = mean(b);
  double std_a = stddev(a);
  double std_b = stddev(b);

  return sum((a - mu_a) % (b - mu_b)) / (std_a * std_b);
}

vec pearson_rho(const mat & A,
                const mat & B){
  assert(size(A) == size(B));
  uint N = A.n_rows;

vec rho = vec(N);
for(uint i = 0; i < N; i++){
rho(i) = pearson_rho(conv_to<vec>::from(A.row(i)),
                       conv_to<vec>::from(B.row(i)));
}
  return rho;
}

void jitter_solve(const TriMesh & mesh,
                 const ProjectiveSolver & solver,
                 const PLCP & ref_plcp,
                 const vec & ref_weights,
                 mat & jitter,
                 mat & noise,
                 uint jitter_rounds){
  uint N = mesh.number_of_spatial_nodes();
  sp_mat P = ref_plcp.P;
  sp_mat U = ref_plcp.U;
  vec q = ref_plcp.q;
  vec ans;
  for(uint j = 0; j < jitter_rounds; j++){      
    cout << "Jitter round: " << j << endl;
    vec perturb = 0.5*(2.0*randu<vec>(N)-1) / (double) N;
    noise.col(j) = perturb;
    q.head(N) = -1.0/(double)N - perturb;
    assert(all(q.head(N) < 0));
      
    vec jitter_q =  P *(P.t() * q);      
    PLCP jitter_plcp = PLCP(P,U,jitter_q,ref_plcp.free_vars);
    
    SolverResult jitter_sol = solver.aug_solve(jitter_plcp);
    jitter.col(j) = jitter_sol.p.head(N);      
  }
}

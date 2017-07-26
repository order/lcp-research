#include "sparse.h"

/*
 * SPARSE ARMA <-> EIGEN CONVERSIONS
 */

using namespace arma;
using namespace Eigen;
using namespace std;

eigen_sp_mat convert_sp_mat_arma_to_eigen(const sp_mat & M){
  clock_t sp_convert_start = clock();

  typedef Triplet<double> T;
  vector<T> tripletList;
  tripletList.reserve(M.n_nonzero);
  for(sp_mat::const_iterator it = M.begin(); it != M.end(); ++it)
    {
      tripletList.push_back(T(it.row(),it.col(),*it));
    }
  eigen_sp_mat eigen_M(M.n_rows,M.n_cols);
  eigen_M.setFromTriplets(tripletList.begin(), tripletList.end());
  double delta_t = (clock() - sp_convert_start)
    / (double)(CLOCKS_PER_SEC);
  cout << "ARMA->EIGEN CONVERT: " << delta_t << "s" << endl;
  eigen_M.makeCompressed();
  return eigen_M;
}

eigen_vec convert_vec_arma_to_eigen(const arma::vec & x){
  clock_t sp_convert_start = clock();

  typedef std::vector<double> stdvec;

  stdvec stdx = conv_to<stdvec>::from(x);
  return eigen_vec::Map(stdx.data(), stdx.size());
}

vec convert_vec_eigen_to_arma(const eigen_vec & x){
  vector<double> v;
  v.resize(x.size());
  VectorXd::Map(&v[0], x.size()) = x;
  return conv_to<vec>::from(v);
}


/*
 * SPARSE SOLVERS
 */

vec sparse_solve(const sp_mat & A, const arma::vec & b, uint mode){
  if(SPARSE_SOLVER_SUPERLU == mode){
    superlu_opts opts;
    opts.equilibrate = true;
    opts.permutation = superlu_opts::COLAMD;
    opts.refine = superlu_opts::REF_NONE;
    return spsolve(A,b,"superlu",opts);
  }

  eigen_sp_mat eigen_A = convert_sp_mat_arma_to_eigen(A);
  eigen_vec eigen_b = convert_vec_arma_to_eigen(b);
  eigen_vec eigen_x(eigen_b.size());
  switch(mode){
  case SPARSE_SOLVER_EIGENLU:
    eigen_x = _sparse_lu_solve(eigen_A, eigen_b); break;
  default:
    cerr << "Sparse solver mode " << mode << " not recognized." << endl;
    assert(false);
    exit(-1);
  }
  return convert_vec_eigen_to_arma(eigen_x);
}

vec _eigen_sparse_solve(const sp_mat & A,
			      const vec & b,
			      eigen_vec (*solver_fn)(const eigen_sp_mat &,
						     const eigen_vec &)){
  eigen_sp_mat eigen_A = convert_sp_mat_arma_to_eigen(A);
  eigen_vec eigen_b = convert_vec_arma_to_eigen(b);
  eigen_vec eigen_x = solver_fn(eigen_A, eigen_b);
  return convert_vec_eigen_to_arma(eigen_x);
}

eigen_vec _sparse_lu_solve(const eigen_sp_mat & A, const eigen_vec & b){
  SparseLU<eigen_sp_mat, COLAMDOrdering<int> > solver;
  solver.compute(A);
  return solver.solve(b);
}

eigen_vec _equilibriate(const eigen_sp_mat & A, const eigen_vec & b,
			eigen_vec (*solver_fn)(const eigen_sp_mat &,
					       const eigen_vec &)){
  Eigen::IterScaling<eigen_sp_mat > scaling;
  eigen_sp_mat scale_A = A;
  scaling.computeRef(scale_A);
  eigen_vec scale_b = scaling.LeftScaling().cwiseProduct(b);
  eigen_vec scale_x = solver_fn(scale_A, scale_b);
  eigen_vec x = scaling.RightScaling().cwiseProduct(scale_x);
  return x;
}





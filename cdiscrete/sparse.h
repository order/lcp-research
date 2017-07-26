#ifndef __Z_EIGEN_SPARSE_INCLUDED__
#define __Z_EIGEN_SPARSE_INCLUDED__

#include <armadillo>
#include <iostream>

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/src/IterativeSolvers/Scaling.h>


#define SPARSE_SOLVER_SUPERLU 0
#define SPARSE_SOLVER_EIGENLU 1
#define SPARSE_SOLVER_BICGSTAB 2
#define SPARSE_SOLVER_GMRES 3

typedef typename Eigen::SparseMatrix<double, Eigen::ColMajor> eigen_sp_mat;
typedef typename Eigen::VectorXd eigen_vec;


eigen_sp_mat convert_sp_mat_arma_to_eigen(const arma::sp_mat & M);
eigen_vec convert_vec_arma_to_eigen(const arma::vec & x);
arma::vec convert_vec_eigen_to_arma(const eigen_vec & x);


arma::vec sparse_solve(const arma::sp_mat & A, const arma::vec & b, uint mode);

arma::vec _eigen_sparse_solve(const arma::sp_mat & A,
			      const arma::vec & b,
			      eigen_vec (*solver_fn)(const eigen_sp_mat &,
					       const eigen_vec &));

eigen_vec _sparse_lu_solve(const eigen_sp_mat & A, const eigen_vec & b);

eigen_vec _equilibriate(const eigen_sp_mat & A, const eigen_vec & b,
			eigen_vec (*solver_fn)(const eigen_sp_mat &,
					       const eigen_vec &));

#endif

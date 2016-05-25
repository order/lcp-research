#ifndef __DATA_IO_INCLUDED__
#define __DATA_IO_INCLUDED__

#include <string>
#include <armadillo>

#include "discrete.h"

using namespace arma;

void import_data(std::string dirname, mat & data, RegGrid & grid);

#endif

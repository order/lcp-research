#ifndef __Z_READER_INCLUDED__
#define __Z_READER_INCLUDED__

#include <armadillo>

#include "discrete.h"
#include "marshaller.h"
#include "mcts.h"
#include "simulate.h"

void read_mcts_config_file(Demarshaller & demarsh,
			   RegGrid & grid,
			   Problem & problem,
			   MCTSContext &context,
			   uint & sim_horizon,
			   mat & start_states,
			   RegGrid & ref_grid,
			   vec & ref_v);

#endif

#include <iostream>
#include "grid.h"

UniformGrid::UniformGrid(vec & low,
			  vec & high,
			 uvec & num_cells) :
  m_low(low),
  m_high(high),
  m_num_cells(num_cells),
  m_num_nodes(num_cells+1),
  m_width((high - low) / num_cells){

  std::cout << m_width << std::endl;
}




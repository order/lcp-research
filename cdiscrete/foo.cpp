#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <armadillo>

#include <assert.h>
#include <boost/algorithm/string.hpp>    
#include <boost/algorithm/string/split.hpp>

using namespace std;
using namespace arma;

mat read_vertices(const string & filename){
  /*
    A pretty brittle parser for INRIA .mesh files
    Extract just the vertex information.
   */
  ifstream fin(filename);
  string line;
  // Find the vertex section
  bool found = false;
  while(getline(fin,line)){
    boost::algorithm::to_lower(line);
    if(std::string::npos != line.find("vertices")){
      found = true;
      break;
    }
  }
  assert(found);
  getline(fin,line);
  uint num_vert = stoul(line);

  mat points = mat(num_vert,3);
  std::vector<std::string> tokens;
  for(uint i; i < num_vert; i++){
    getline(fin,line);
    cout << line << endl;
    boost::trim(line);
    boost::split(tokens, line, boost::is_any_of(" \t"),boost::token_compress_on);
    cout << tokens.size() << endl;
    assert(4 == tokens.size());
    for(uint j = 0; j < 3; ++j){
      points(i,j) = stod(tokens[j]);
    }
  }
  return points;
}

int main(int argc, char** argv)
{  
  mat vertices = read_vertices("test.mesh");
  cout << vertices;
}

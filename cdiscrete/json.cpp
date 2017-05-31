#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

/*JSON parsing routines*/


typedef std::map<std::string,std::string> ParamMap;

void print_property_tree(boost::property_tree::ptree const& pt)
{
  /*
    Print out an entire property tree.
  */
  using boost::property_tree::ptree;
  ptree::const_iterator end = pt.end();
  for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
    std::cout << it->first << ": " << it->second.get_value<std::string>() << std::endl;
    print_property_tree(it->second);
  }
}


ParamMap & json_file_to_param_map(const std::string & filename){
  using boost::property_tree::ptree;
  using boost::property_tree::read_json;

  ptree prop_tree;
  std::ifstream str_stream(filename);
  read_json(str_stream, prop_tree);

  ParamMap parameters;
  
  for(ptree::const_iterator it = prop_tree.begin(); it != end; ++it){
    parameters.insert(it->first, it->second.get_value<std::string>());
  }
  return parameters;
}

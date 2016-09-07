#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

using namespace std;

int main(int argc, char** argv)
{  
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("compression", po::value<int>()->default_value(10), "set compression level")
    ("file", po::value<string>()->default_value("foo.dat"), "filename")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  if (vm.count("compression")) {
    cout << "Compression level was set to " 
         << vm["compression"].as<int>() << ".\n";
  } else {
    cout << "Compression level was not set.\n";
  }
  if (vm.count("file")) {
    cout << "Filename is " 
         << vm["file"].as<string>() << ".\n";
  } else {
    cout << "Filename was not set.\n";
  }
  
}

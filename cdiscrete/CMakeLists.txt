# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

project( mesh )


cmake_minimum_required(VERSION 2.8.11)

# CCACHE
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()
project( mesh )


# Explicitly set some flags
set(CMAKE_CXX_FLAGS "-O3 -fstack-protector -Wformat -Werror=format-security -frounding-math -FORTIFY_SOURCE=1 -fext-numeric-literals -std=c++11")

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )
  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  
endif()

# include helper file
include( ${CGAL_USE_FILE} )


# Boost and its components
find_package( Boost COMPONENTS program_options filesystem REQUIRED )
if ( NOT Boost_FOUND )
  message(STATUS "This project requires the Boost library, and will not be compiled.")
  return()  
endif()

#find_package(Eigen3 REQUIRED)
find_package(LibArchive REQUIRED)
find_package(Armadillo REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "Manually configuring SuperLU")
set(SUPERLU_INCLUDE_DIR /usr/local/include/superlu)
set(SUPERLU_LIBRARIES /usr/local/lib/libsuperlu.so)
message(STATUS "Manually configuring OpenBLAS")
set(OPENBLAS_INCLUDE_DIR /opt/OpenBLAS/include/openblas)
set(OPENBLAS_LIBRARIES /opt/OpenBLAS/lib/libopenblas.so)

include_directories( ${Boost_INCLUDE_DIR} )
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${LibArchive_INCLUDE_DIRS})
include_directories(${LAPACK_INCLUDE_DIR})
#include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${SUPERLU_INCLUDE_DIR})
include_directories(${OPENBLAS_INCLUDE_DIR})


# Include for local directory

# include for local package


# Creating entries for all C++ files with "main" routine
# ##########################################################

include( CGAL_CreateSingleSourceCGALProgram )
list(APPEND COMMON_FILES "misc.cpp" "io.cpp" "simulator.cpp"
  "lcp.cpp" "discretizer.cpp" "solver.cpp" "basis.cpp")

# create_single_source_cgal_program("di_basis.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp"
#   ${COMMON_FILES})
# create_single_source_cgal_program("di_refine.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp"
#   ${COMMON_FILES})
# create_single_source_cgal_program("di_approx.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp"
#   ${COMMON_FILES})
# create_single_source_cgal_program("di_foo.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp"
#   ${COMMON_FILES})
# create_single_source_cgal_program("di_single.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp"
#   ${COMMON_FILES})
# create_single_source_cgal_program("di_grid_approx.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp" 
#   ${COMMON_FILES})
# create_single_source_cgal_program("di_bw_origin_approx.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp" 
#   ${COMMON_FILES})
# create_single_source_cgal_program("di_iso_bw_origin_approx.cpp" "tri_mesh.cpp" "di.cpp"
#   "refine.cpp" "solver.cpp" "smooth.cpp"
#   ${COMMON_FILES})
#create_single_source_cgal_program("hillcar_gen.cpp" "tri_mesh.cpp"
#  "hillcar.cpp" ${COMMON_FILES})
#create_single_source_cgal_program("hillcar_refine.cpp" "tri_mesh.cpp"
#  "hillcar.cpp" "refine.cpp" ${COMMON_FILES})

#create_single_source_cgal_program("minop_approx.cpp" "tri_mesh.cpp"
#  "minop.cpp" ${COMMON_FILES})
#create_single_source_cgal_program("minop_sensitivity.cpp" "tri_mesh.cpp"
#  ${COMMON_FILES})
#create_single_source_cgal_program("minop_free_vs_bound.cpp" "tri_mesh.cpp"
#  ${COMMON_FILES})
#create_single_source_cgal_program("minop_flow_refine.cpp" "tri_mesh.cpp"
#  "minop.cpp" ${COMMON_FILES})

#create_single_source_cgal_program("basis_gen.cpp" "tri_mesh.cpp"
#  ${COMMON_FILES})

#create_single_source_cgal_program("dubins_roundabout.cpp" "tet_mesh.cpp" "dubins.cpp"
#  ${COMMON_FILES})
#create_single_source_cgal_program("dubins_refine.cpp" "tet_mesh.cpp" "dubins.cpp"
#  "refine.cpp" ${COMMON_FILES})
#create_single_source_cgal_program("tet_interp.cpp" "tet_mesh.cpp" ${COMMON_FILES})

#add_executable (hallway_approx hallway_approx.cpp hallway.cpp ${COMMON_FILES})

add_executable (grid_test grid_test.cpp grid.cpp misc.cpp discretizer.cpp points.cpp)


list(APPEND CDISCRETE_LIBRARIES
        ${ARMADILLO_LIBRARIES}
        ${LibArchive_LIBRARIES}
        ${Boost_LIBRARIES}
        ${EIGEN3_LIBRARIES}
        ${SUPERLU_LIBRARIES}
        ${OPENBLAS_LIBRARIES}
        ${LAPACK_LIBRARIES})
message(STATUS "Libraries: ${CDISCRETE_LIBRARIES}")

#target_link_libraries(di_refine ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_foo ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_basis ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_single ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_refine ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_approx ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_grid_approx ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_bw_origin_approx ${CDISCRETE_LIBRARIES})
#target_link_libraries(di_iso_bw_origin_approx ${CDISCRETE_LIBRARIES})

#target_link_libraries(hillcar_gen ${CDISCRETE_LIBRARIES})
#target_link_libraries(hillcar_refine ${CDISCRETE_LIBRARIES})

#target_link_libraries(min_gen ${CDISCRETE_LIBRARIES})
#target_link_libraries(min_refine ${CDISCRETE_LIBRARIES})
#target_link_libraries(minop_approx ${CDISCRETE_LIBRARIES})
#target_link_libraries(minop_sensitivity ${CDISCRETE_LIBRARIES})
#target_link_libraries(minop_free_vs_bound ${CDISCRETE_LIBRARIES})
#target_link_libraries(minop_flow_refine ${CDISCRETE_LIBRARIES})


#target_link_libraries(basis_gen ${CDISCRETE_LIBRARIES})

#target_link_libraries(dubins_roundabout ${CDISCRETE_LIBRARIES})
#target_link_libraries(dubins_refine ${CDISCRETE_LIBRARIES})

#target_link_libraries(tet_interp ${CDISCRETE_LIBRARIES})
#target_link_libraries(hallway_approx ${CDISCRETE_LIBRARIES})

target_link_libraries(grid_test  ${CDISCRETE_LIBRARIES})

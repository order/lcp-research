# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

project( mesh )


cmake_minimum_required(VERSION 2.8.11)

# CCACHE
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()
project( mesh )

# Use C++ 11
set (CMAKE_CXX_STANDARD 11)

# Explicitly set some flags
set(CMAKE_CXX_FLAGS "-O -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2  -frounding-math")

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


find_package(LibArchive REQUIRED)
find_package(Armadillo REQUIRED)
include_directories( ${Boost_INCLUDE_DIR} )
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${LibArchive_INCLUDE_DIRS})


# Include for local directory

# include for local package


# Creating entries for all C++ files with "main" routine
# ##########################################################

include( CGAL_CreateSingleSourceCGALProgram )
list(APPEND COMMON_FILES "misc.cpp" "io.cpp" "simulator.cpp"
  "lcp.cpp" "discretizer.cpp")
create_single_source_cgal_program("di_gen.cpp" "tri_mesh.cpp" "di.cpp"
  ${COMMON_FILES})
create_single_source_cgal_program("di_refine.cpp" "tri_mesh.cpp"
  "di.cpp" "refine.cpp"
  ${COMMON_FILES})        
create_single_source_cgal_program("hillcar_gen.cpp" "tri_mesh.cpp"
  "hillcar.cpp" ${COMMON_FILES})
create_single_source_cgal_program("hillcar_refine.cpp" "tri_mesh.cpp"
  "hillcar.cpp" "refine.cpp" ${COMMON_FILES})  

create_single_source_cgal_program("dubins_gen.cpp" "tet_mesh.cpp" "dubins.cpp"
  ${COMMON_FILES})
create_single_source_cgal_program("dubins_refine.cpp" "tet_mesh.cpp" "dubins.cpp"
  "refine.cpp" ${COMMON_FILES})
create_single_source_cgal_program("tet_interp.cpp" "tet_mesh.cpp" ${COMMON_FILES})

list(APPEND CDISCRETE_LIBRARIES
        ${ARMADILLO_LIBRARIES}
        ${LibArchive_LIBRARIES}
        ${Boost_LIBRARIES})

target_link_libraries(di_gen ${CDISCRETE_LIBRARIES})
target_link_libraries(di_refine ${CDISCRETE_LIBRARIES})
target_link_libraries(hillcar_gen ${CDISCRETE_LIBRARIES})
target_link_libraries(hillcar_refine ${CDISCRETE_LIBRARIES})

target_link_libraries(dubins_gen ${CDISCRETE_LIBRARIES})
target_link_libraries(dubins_refine ${CDISCRETE_LIBRARIES})
target_link_libraries(tet_interp ${CDISCRETE_LIBRARIES})

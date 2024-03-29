cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
PROJECT(rbfnetwork2)

SET(CMAKE_VERBOSE_MAKEFILE OFF)

SET(CMAKE_BUILD_TYPE distribution)
#SET(CMAKE_BUILD_TYPE debug)


SET(CMAKE_CXX_FLAGS_DISTRIBUTION " -Wall -O3 -Wno-unused-variable -D_POSIX_PTHREAD_SEMANTICS -D_REENTRANT -DNDEBUG -DBOOST_UBLAS_NDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG " -Wall -g3  -D_POSIX_PTHREAD_SEMANTICS -D_REENTRANT -DBZ_DEBUG")

#boost bits
find_package( Boost COMPONENTS python REQUIRED)
find_package( PythonLibs REQUIRED)

ADD_DEFINITIONS(-Wall -Wno-deprecated)
include_directories(${Boost_INCLUDES} ${PYTHON_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}/numpy)

ADD_LIBRARY(functions SHARED rbfnetwork/functions.cpp)

TARGET_LINK_LIBRARIES(functions ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} lapack)

cmake_minimum_required(VERSION 3.10)
project(cppban)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)


# Imposta il compilatore e le opzioni di compilazione
set(CMAKE_CXX_COMPILER "/home/tesi1/c/rel/bin/clang++")

# vettorizzazione on
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target ve-linux -O3")
# vettorizzazione off
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target ve-linux -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target ve-linux -fno-vectorize -fno-slp-vectorize")


# Init message
message("Build profile: "  ${CMAKE_BUILD_TYPE})

# Ban Library
add_subdirectory(src)

# Add any additional targets here, such as executables or tests
#add_subdirectory(examples)
add_subdirectory(apps)
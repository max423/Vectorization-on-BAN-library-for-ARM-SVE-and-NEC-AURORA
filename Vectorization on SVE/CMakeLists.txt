cmake_minimum_required(VERSION 3.10)
project(cppban)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

set(CMAKE_CXX_COMPILER "armclang++")

# Aggiungi l'opzione di compilazione per SVE
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+sve -O3")

# NO vettorizzazione 
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+sve -O0")
# SI vettorizzazione
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+sve -O3")


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Rpass-missed=inline")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-vectorize")


# Init message
message("Build profile: "  ${CMAKE_BUILD_TYPE})
message("Compiler profile: "  ${CMAKE_CXX_COMPILER})
# Ban Library
add_subdirectory(src)

# Add any additional targets here, such as executables or tests
#add_subdirectory(examples)
add_subdirectory(apps)
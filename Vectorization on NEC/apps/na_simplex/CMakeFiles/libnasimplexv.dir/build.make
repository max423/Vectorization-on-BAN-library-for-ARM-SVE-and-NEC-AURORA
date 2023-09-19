# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tesi1/R3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tesi1/R3

# Include any dependencies generated for this target.
include apps/na_simplex/CMakeFiles/libnasimplexv.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include apps/na_simplex/CMakeFiles/libnasimplexv.dir/compiler_depend.make

# Include the progress variables for this target.
include apps/na_simplex/CMakeFiles/libnasimplexv.dir/progress.make

# Include the compile flags for this target's objects.
include apps/na_simplex/CMakeFiles/libnasimplexv.dir/flags.make

apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o: apps/na_simplex/CMakeFiles/libnasimplexv.dir/flags.make
apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o: apps/na_simplex/src/NA_Simplex.cpp
apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o: apps/na_simplex/CMakeFiles/libnasimplexv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tesi1/R3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o"
	cd /home/tesi1/R3/apps/na_simplex && /home/tesi1/c/rel/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o -MF CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o.d -o CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o -c /home/tesi1/R3/apps/na_simplex/src/NA_Simplex.cpp

apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.i"
	cd /home/tesi1/R3/apps/na_simplex && /home/tesi1/c/rel/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tesi1/R3/apps/na_simplex/src/NA_Simplex.cpp > CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.i

apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.s"
	cd /home/tesi1/R3/apps/na_simplex && /home/tesi1/c/rel/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tesi1/R3/apps/na_simplex/src/NA_Simplex.cpp -o CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.s

# Object files for target libnasimplexv
libnasimplexv_OBJECTS = \
"CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o"

# External object files for target libnasimplexv
libnasimplexv_EXTERNAL_OBJECTS =

apps/na_simplex/liblibnasimplexv.a: apps/na_simplex/CMakeFiles/libnasimplexv.dir/src/NA_Simplex.cpp.o
apps/na_simplex/liblibnasimplexv.a: apps/na_simplex/CMakeFiles/libnasimplexv.dir/build.make
apps/na_simplex/liblibnasimplexv.a: apps/na_simplex/CMakeFiles/libnasimplexv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tesi1/R3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library liblibnasimplexv.a"
	cd /home/tesi1/R3/apps/na_simplex && $(CMAKE_COMMAND) -P CMakeFiles/libnasimplexv.dir/cmake_clean_target.cmake
	cd /home/tesi1/R3/apps/na_simplex && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libnasimplexv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/na_simplex/CMakeFiles/libnasimplexv.dir/build: apps/na_simplex/liblibnasimplexv.a
.PHONY : apps/na_simplex/CMakeFiles/libnasimplexv.dir/build

apps/na_simplex/CMakeFiles/libnasimplexv.dir/clean:
	cd /home/tesi1/R3/apps/na_simplex && $(CMAKE_COMMAND) -P CMakeFiles/libnasimplexv.dir/cmake_clean.cmake
.PHONY : apps/na_simplex/CMakeFiles/libnasimplexv.dir/clean

apps/na_simplex/CMakeFiles/libnasimplexv.dir/depend:
	cd /home/tesi1/R3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tesi1/R3 /home/tesi1/R3/apps/na_simplex /home/tesi1/R3 /home/tesi1/R3/apps/na_simplex /home/tesi1/R3/apps/na_simplex/CMakeFiles/libnasimplexv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/na_simplex/CMakeFiles/libnasimplexv.dir/depend

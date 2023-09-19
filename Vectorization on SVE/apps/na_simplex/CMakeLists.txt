set(APPFLAGS -Wno-deprecated-declarations -ffast-math)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_library(libnasimplex src/NA_Simplex.cpp)
target_link_libraries(libnasimplex PRIVATE cppban Eigen3::Eigen)
target_compile_options(libnasimplex PRIVATE ${APPFLAGS})
target_include_directories(libnasimplex PRIVATE ./include ${PROJECT_SOURCE_DIR}/include/cppban)

add_library(libnasimplexv src/NA_Simplex.cpp)
target_link_libraries(libnasimplexv PRIVATE cppbanv Eigen3::Eigen)
target_compile_options(libnasimplexv PRIVATE ${APPFLAGS})
target_include_directories(libnasimplexv PRIVATE ./include ${PROJECT_SOURCE_DIR}/include/cppban)

add_library(libnasimplexvalidate src/NA_Simplex_validate.cpp)
target_link_libraries(libnasimplexvalidate PRIVATE cppban Eigen3::Eigen)
target_compile_options(libnasimplexvalidate PRIVATE ${APPFLAGS})
target_include_directories(libnasimplexvalidate PRIVATE ./include ${PROJECT_SOURCE_DIR}/include/cppban)

add_library(libnasimplexvalidatev src/NA_Simplex_validate.cpp)
target_link_libraries(libnasimplexvalidatev PRIVATE cppbanv Eigen3::Eigen)
target_compile_options(libnasimplexvalidatev PRIVATE ${APPFLAGS})
target_include_directories(libnasimplexvalidatev PRIVATE ./include ${PROJECT_SOURCE_DIR}/include/cppban)


add_executable(run_simplex run_simplex.cpp)
target_link_libraries(run_simplex PRIVATE libnasimplex cppban Eigen3::Eigen)
target_include_directories(run_simplex PRIVATE ./include ${PROJECT_SOURCE_DIR}/include/cppban)

add_executable(run_simplexv run_simplex.cpp)
target_link_libraries(run_simplexv PRIVATE libnasimplexv cppbanv Eigen3::Eigen)
target_include_directories(run_simplexv PRIVATE ./include ${PROJECT_SOURCE_DIR}/include/cppban)

add_executable(validate_simplex run_simplex.cpp)
target_link_libraries(validate_simplex PRIVATE libnasimplexvalidate cppban Eigen3::Eigen)
target_include_directories(validate_simplex PRIVATE ./include ${PROJECT_SOURCE_DIR}/include/cppban)

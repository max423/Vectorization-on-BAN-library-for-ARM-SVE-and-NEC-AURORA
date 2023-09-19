add_executable(basic basic.cpp)
target_link_libraries(basic PRIVATE cppban)
target_include_directories(basic PRIVATE ${PROJECT_SOURCE_DIR}/include/cppban)
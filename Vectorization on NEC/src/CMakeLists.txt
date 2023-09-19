set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

add_library(cppban
    ban.cpp
)

add_library(cppbanv
    banv.cpp
)

target_include_directories(cppban PUBLIC ${PROJECT_SOURCE_DIR}/include/cppban)
target_compile_options(cppban PRIVATE ${LIBFLAGS})

target_include_directories(cppbanv PUBLIC ${PROJECT_SOURCE_DIR}/include/cppban)
target_compile_options(cppbanv PRIVATE ${LIBFLAGS})

install(TARGETS cppban
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
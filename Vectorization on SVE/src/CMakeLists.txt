#set(LIBFLAGS -Wno-deprecated-declarations -ffast-math -O3)
#set(LIBFLAGS -O3 -Rpass-missed=vectorize)

add_library(cppban
    ban.cpp
)

add_library(cppbanv
        banv.cpp
)

target_include_directories(cppban PUBLIC ${PROJECT_SOURCE_DIR}/include/cppban)
target_compile_options(cppban PRIVATE -Wno-deprecated-declarations ${LIBFLAGS})

target_include_directories(cppbanv PUBLIC ${PROJECT_SOURCE_DIR}/include/cppban)
target_compile_options(cppbanv PRIVATE -Wno-deprecated-declarations ${LIBFLAGS})




#target_include_directories(cppban PUBLIC ${PROJECT_SOURCE_DIR}/include/cppban)
#target_compile_options(cppban PRIVATE -Rpass-missed=slp-vectorizer ${LIBFLAGS})
#
#target_include_directories(cppbanv PUBLIC ${PROJECT_SOURCE_DIR}/include/cppban)
#target_compile_options(cppbanv PRIVATE -Rpass-missed=slp-vectorizer ${LIBFLAGS})

install(TARGETS cppban
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

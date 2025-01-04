file(GLOB SOURCES
    src/*.cpp
    src/**/*.cpp
)
add_library(DumbTorch STATIC ${SOURCES})

target_include_directories(DumbTorch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET DumbTorch PROPERTY CXX_STANDARD 20)
endif()

# add_subdirectory(tests)
# add_subdirectory(python_bindings)

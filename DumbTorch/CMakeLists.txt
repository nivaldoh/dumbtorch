# Used Python on path with "pip install pybind11" and then ran "python -m pybind11 --cmakedir"
# TODO: don't hardcode this
set(pybind11_DIR "C:/Users/niv/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11")

find_package(CUDA REQUIRED)

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.cu"
)

# Create the main library
add_library(dumbtorch STATIC ${SOURCES})

# Add include directories
target_include_directories(dumbtorch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link CUDA
target_link_libraries(dumbtorch PRIVATE ${CUDA_LIBRARIES})

# Optional: Create an executable for testing/development
add_executable(dumbtorch_app "src/DumbTorch.cpp")
target_link_libraries(dumbtorch_app PRIVATE dumbtorch)

# Python bindings (if enabled)
#if(DUMBTORCH_BUILD_PYTHON)
    #find_package(Python COMPONENTS Interpreter Development REQUIRED)
    #find_package(pybind11 CONFIG REQUIRED)
#    
    #pybind11_add_module(_C MODULE
        #python/dumbtorch/_C.cpp
    #)
#    
    #target_link_libraries(_C PRIVATE dumbtorch)
#endif()

# Tests (if enabled)
if(DUMBTORCH_BUILD_TESTS)
    add_subdirectory(tests)
endif()
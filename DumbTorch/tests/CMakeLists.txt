# Enable testing
# enable_testing()

# Use FetchContent for GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Collect test source files
#file(GLOB_RECURSE TEST_SOURCES 
    #"*.cpp"
    #"*.cu"
#)

# Collect test source files - be explicit about where we look
file(GLOB_RECURSE TEST_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cu"
)
# Print what we found (for debugging)
message(STATUS "Test sources: ${TEST_SOURCES}")

# Create test executable
add_executable(dumbtorch_tests ${TEST_SOURCES})

# Link against the main library and testing framework
target_link_libraries(dumbtorch_tests 
    PRIVATE 
        dumbtorch
        ${CUDA_LIBRARIES}
        GTest::gtest_main
)

# Add include directories
target_include_directories(dumbtorch_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Use GoogleTest module to discover tests
include(GoogleTest)
gtest_discover_tests(dumbtorch_tests)
cmake_minimum_required(VERSION 3.20)

project(LearningGtest)

# setup
set(CMAKE_CXX_STANDARD 11)



# Add Directories
add_subdirectory(src)
add_subdirectory(tests)

# fetch libraries
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
    # For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


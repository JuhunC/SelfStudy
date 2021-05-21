# Google Test

## Prerequisite(Windows)
> Install [Cmake](https://cmake.org/), make sure to add PATH to your system

### CMake System
* CMake system is configured by *CMakelists.txt"


## CMakelists.txt
* ### Set CMake Version
  Always needs to be on top of the *CMakelists.txt* file.
```
  cmake_minimum_required(VERSION 3.20)
```
* ### Set Project Name
```
  project(LearningGtest)
```
* ### Gtest Require Minimum C++11 
```
  set(CMAKE_CXX_STANDARD 11)
```

* ### Add Subdirectories
  Add Subdirectories that has other *CMakelists.txt*. Both **[src],[test]** directory ***MUST*** have *CMakelists.txt*!!!
```
  add_subdirectory(src)
  add_subdirectory(tests)
```

* ### Fetching Libraries for GTEST
```
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
    # For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
```
* ### Enable Testing
  Enable Testing all directories below.
```
  enable_testing()
```
* ### Add Executable Files for Compilation
  map a Name to a binary file, which was compiled from multiple files (*hello_test.cc ...*).
```
  add_executable(HELLO_TEST hello_test.cc)
  add_executable(TESTs module.cpp module.h main.cpp)
```

* ### Link Libraries
  HelloTest (mapped name) is Linked to gtest_main.
```
  target_link_libraries(
    HelloTest
    gtest_main
  )
```

## Run Gtest
```
  cmake -S $(PROJECT_DIR)/ -B $(PROJECT_DIR)/build
  cmake --build $(PROJECT_DIR)/build
```
* Test Execution file is located at **$(PROJECT_DIR)/build/test/$(Mapped Name).exe**
* You can create multiple Gtest execution files using Different proejcts
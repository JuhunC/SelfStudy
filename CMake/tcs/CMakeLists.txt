enable_testing()
project(${PROJECT_NAME}_TEST)

add_executable(${PROJECT_NAME}_TEST
  gtest_tc.cpp
)
target_link_libraries(${PROJECT_NAME}_TEST
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_TEST)
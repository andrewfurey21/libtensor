cmake_minimum_required(VERSION 3.10)
project(libtensor)
find_package(GTest REQUIRED)
include(CTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB LIB_SOURCES "src/*.c")

add_library(tensor STATIC ${LIB_SOURCES})

target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB EXAMPLE_SOURCES "examples/*.c")

foreach(example_source ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_source} NAME_WE)
    add_executable(${example_name} ${example_source})
    target_link_libraries(${example_name} PRIVATE tensor m)
endforeach()

file(GLOB TEST_SOURCES "tests/*.cpp")

foreach(test ${TEST_SOURCES})
    get_filename_component(test_name ${test} NAME_WE)
    add_executable(${test_name} ${test})
    target_link_libraries(${test_name}
        PRIVATE 
        GTest::gtest_main
        pthread
        tensor
        m
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()


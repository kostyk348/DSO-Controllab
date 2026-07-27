cmake_minimum_required(VERSION 3.16)
project(dso_gp C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# OpenMP for parallel fitness evaluation
find_package(OpenMP QUIET)

add_library(gp_core STATIC
    plant.c
    gp_tree.c
    gp_evolve.c
    gp_export.c
)

target_include_directories(gp_core PUBLIC .)
if(OpenMP_C_FOUND)
    target_compile_options(gp_core PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(gp_core PUBLIC OpenMP::OpenMP_C)
endif()

add_executable(dso_gp main.c)
target_link_libraries(dso_gp PRIVATE gp_core m)
target_compile_options(dso_gp PRIVATE -Wall -Wextra -Wpedantic)

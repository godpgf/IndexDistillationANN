cmake_minimum_required(VERSION 3.5.0)
project(VastKMeans VERSION 0.1.0 LANGUAGES C CXX)
include(CheckCXXCompilerFlag)
include (FetchContent)

#----------------------------------------------------------
# 1. 依赖：jemalloc
#----------------------------------------------------------
find_program(JEMALLOC_CONFIG_EXECUTABLE jemalloc-config)
if(JEMALLOC_CONFIG_EXECUTABLE)
    execute_process(COMMAND ${JEMALLOC_CONFIG_EXECUTABLE} --libdir
                    OUTPUT_VARIABLE JEMALLOCLD
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_library(JEMALLOC_LIB jemalloc PATHS ${JEMALLOCLD} NO_DEFAULT_PATH)
endif()

if(JEMALLOC_LIB)
    add_library(jemalloc INTERFACE)
    target_link_libraries(jemalloc INTERFACE ${JEMALLOC_LIB})
    target_link_directories(jemalloc INTERFACE ${JEMALLOCLD})
    message(STATUS "jemalloc found: ${JEMALLOC_LIB}")
else()
    add_library(jemalloc INTERFACE)   # 空接口，什么都不加
    message(STATUS "jemalloc NOT found, disabled")
endif()

#----------------------------------------------------------
# 2. 后端选择
#----------------------------------------------------------
set(PARLAY_BACKEND "PBB" CACHE STRING "Backend: OPENMP | CILK | PBB")
set_property(CACHE PARLAY_BACKEND PROPERTY STRINGS OPENMP CILK PBB)

# 统一编译标志
# -DSTATS -DHOMEGROWN -pthread -mcx16 -O3 -std=c++17
set(COMMON_FLAGS
    -DSTATS -DHOMEGROWN -pthread -mcx16 -O3 -std=c++17 -march=native -DNDEBUG
)

if(PARLAY_BACKEND STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    set(EXTRA_DEF   PARLAY_OPENMP)
    set(EXTRA_LIBS  OpenMP::OpenMP_CXX)

elseif(PARLAY_BACKEND STREQUAL "CILK")
    # 需要编译器支持 cilkplus（GCC 5~7 时代才有）
    set(CMAKE_CXX_COMPILER g++)   # 强制 g++
    set(EXTRA_DEF   PARLAY_CILK)
    set(EXTRA_FLAGS -fcilkplus)
    set(EXTRA_LIBS  cilkrts)

elseif(PARLAY_BACKEND STREQUAL "PBB")
    set(EXTRA_DEF   HOMEGROWN)
    set(EXTRA_FLAGS -pthread)
    set(EXTRA_LIBS  pthread)

else()
    message(FATAL_ERROR "Unknown PARLAY_BACKEND: ${PARLAY_BACKEND}")
endif()

find_package(Threads)

# FetchContent_Declare(parlaylib
#   GIT_REPOSITORY  https://github.com/cmuparlay/parlaylib.git
#   GIT_TAG         master
# )
# FetchContent_GetProperties(parlaylib)
# if(NOT parlaylib_POPULATED)
#   FetchContent_Populate(parlaylib)  
#   add_subdirectory(${parlaylib_SOURCE_DIR} EXCLUDE_FROM_ALL)
# endif()
# 计算绝对路径
set(parlaylib_root "${CMAKE_CURRENT_SOURCE_DIR}/parlaylib")
# 确保目录存在
if(NOT EXISTS "${parlaylib_root}/CMakeLists.txt")
  message(FATAL_ERROR "parlaylib 源码目录不存在：${parlaylib_root}")
endif()
# 直接把它加进来
add_subdirectory(${parlaylib_root} EXCLUDE_FROM_ALL)

# add_executable(HNSW ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/bench/neighborsTime.C)
# target_compile_options(HNSW PRIVATE "-include" "neighbors.h")
# target_include_directories(HNSW
    # PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/HNSW
    # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# )
# target_link_libraries(HNSW PRIVATE parlay)

add_executable(PQ ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/PQ/pq.cpp)
target_compile_options(PQ PRIVATE -O3 -std=c++17 -march=native -mcx16)
target_compile_definitions(PQ PRIVATE STATS HOMEGROWN NDEBUG)
target_include_directories(PQ
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(PQ PRIVATE parlay Threads::Threads dl)

# add_executable(TestPQ ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/PQ/test.cpp)
# target_include_directories(TestPQ PRIVATE -O3 -std=c++17 -march=native -mcx16
    # PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    # PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# )
# target_link_libraries(TestPQ PRIVATE parlay)

add_executable(ExpPQDict ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/PQ/exportDict.cpp)
target_compile_options(ExpPQDict PRIVATE -O3 -std=c++17 -march=native -mcx16)
target_compile_definitions(ExpPQDict PRIVATE STATS HOMEGROWN NDEBUG)
target_include_directories(ExpPQDict
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(ExpPQDict PRIVATE parlay Threads::Threads dl)

add_executable(MultiStart ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/bench/neighborsTime.C)
target_compile_options(MultiStart PRIVATE -O3 -std=c++17 -march=native -mcx16 "-include" "neighbors.h")
target_compile_definitions(MultiStart PRIVATE STATS HOMEGROWN NDEBUG)
target_include_directories(MultiStart
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/Multi-Start
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(MultiStart PRIVATE parlay Threads::Threads dl)

# export sp
add_executable(ExpSP ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/PQ/exportSP.cpp)
target_compile_options(ExpSP PRIVATE -O3 -std=c++17 -march=native -mcx16)
target_compile_definitions(ExpSP PRIVATE STATS HOMEGROWN NDEBUG)
target_include_directories(ExpSP
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(ExpSP PRIVATE parlay Threads::Threads dl)

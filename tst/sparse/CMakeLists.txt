add_executable(sparse_storage_test storage.cpp)
add_executable(sparse_benchmark benchmark.cpp)
add_executable(sparse_advection_test advection.cpp)
foreach(target sparse_storage_test sparse_benchmark sparse_advection_test)
  target_link_libraries(${target} PRIVATE athena_sparse)
endforeach()

add_test(NAME sparse_storage COMMAND sparse_storage_test)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME sparse_device_capacity COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/expect_capacity_failure.py $<TARGET_FILE:sparse_storage_test>)
set_tests_properties(sparse_device_capacity PROPERTIES TIMEOUT 60)
add_test(NAME sparse_advection_periodic COMMAND sparse_advection_test
  --blocks 8 --cells 16 --fields 13 --steps 256 --courant 1)
add_test(NAME sparse_advection_reverse COMMAND sparse_advection_test
  --blocks 7 --cells 12 --fields 9 --steps 168 --courant 1 --direction -1)
add_test(NAME sparse_advection_diffusive COMMAND sparse_advection_test
  --blocks 8 --cells 16 --fields 13 --steps 40 --courant 0.4)
add_test(NAME sparse_advection_no_prune COMMAND sparse_advection_test
  --blocks 8 --cells 16 --fields 9 --steps 40 --courant 0.7 --prune 0)
add_test(NAME sparse_advection_one_block COMMAND sparse_advection_test
  --blocks 1 --cells 32 --fields 4 --steps 64 --courant 1)
add_test(NAME sparse_benchmark_empty COMMAND sparse_benchmark
  --blocks 8 --cells 8 --fields 7 --fraction 0 --iterations 5 --repeats 1)
add_test(NAME sparse_benchmark_partial COMMAND sparse_benchmark
  --blocks 16 --cells 16 --fields 17 --fraction 0.25 --iterations 8 --repeats 1)
add_test(NAME sparse_benchmark_full COMMAND sparse_benchmark
  --blocks 8 --cells 8 --fields 7 --fraction 1 --iterations 5 --repeats 1)

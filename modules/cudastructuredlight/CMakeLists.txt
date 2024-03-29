if(IOS OR  WINRT OR (NOT HAVE_CUDA AND NOT WITH_EIGEN AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudastructuredlight)
endif()

set(the_description "CUDA-accelerated Structured Light API")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)
ocv_module_include_directories(cudastructuredlight ${EIGEN3_INCLUDE_DIRS})
ocv_define_module(cudastructuredlight opencv_core opencv_structured_light opencv_cudev opencv_cudaarithm  WRAP python)

if(IOS OR  WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudastereo)
endif()

set(the_description "CUDA-accelerated Stereo Correspondence")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  ocv_module_include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
ocv_define_module(cudastereo opencv_3d opencv_stereo OPTIONAL opencv_cudev WRAP python)

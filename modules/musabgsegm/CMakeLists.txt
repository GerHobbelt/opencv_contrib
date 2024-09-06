if(IOS OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
  ocv_module_disable(musabgsegm)
endif()

set(the_description "MUSA-accelerated Background Segmentation")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

ocv_define_module(musabgsegm opencv_video WRAP python)

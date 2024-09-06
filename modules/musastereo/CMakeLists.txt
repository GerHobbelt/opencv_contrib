if(IOS OR  WINRT OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
  ocv_module_disable(musastereo)
endif()

set(the_description "MUSA-accelerated Stereo Correspondence")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

ocv_define_module(musastereo opencv_calib3d OPTIONAL opencv_mudev WRAP python)

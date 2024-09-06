if(IOS OR WINRT OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
  ocv_module_disable(musawarping)
endif()

set(the_description "MUSA-accelerated Image Warping")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow -Wdeprecated-copy -Wdeprecated-declarations -Wmissing-prototypes -Wmacro-redefined)

ocv_define_module(musawarping opencv_core opencv_imgproc OPTIONAL opencv_mudev WRAP python)

if(IOS OR WINRT OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
  ocv_module_disable(musafilters)
endif()

set(the_description "MUSA-accelerated Image Filtering")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow -Wmissing-prototypes -Wdeprecated-copy -Wsign-compare -Wabsolute-value -Wunused-function -Wmacro-redefined)

ocv_define_module(musafilters opencv_imgproc opencv_musaarithm WRAP python)

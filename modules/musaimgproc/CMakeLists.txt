# if(IOS OR WINRT OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
if(IOS OR WINRT OR (NOT HAVE_MUSA))
  ocv_module_disable(musaimgproc)
endif()

set(the_description "MUSA-accelerated Image Processing")


ocv_define_module(musaimgproc opencv_imgproc OPTIONAL opencv_mudev opencv_musaarithm opencv_musafilters WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

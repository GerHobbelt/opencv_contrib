if(IOS OR WINRT OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
  ocv_module_disable(musafeatures2d)
endif()

set(the_description "MUSA-accelerated Feature Detection and Description")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter -Wshadow -Wdeprecated-copy -Wmissing-prototypes -Wsign-compare -Wunused-const-variable -Wunused-function -Wmacro-redefined)

ocv_define_module(musafeatures2d opencv_features2d opencv_musafilters opencv_musawarping WRAP python)

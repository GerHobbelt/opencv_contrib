if(NOT HAVE_MUSA)
  ocv_module_disable(musalegacy)
endif()

set(the_description "MUSA-accelerated Computer Vision (legacy)")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4130 /wd4324 /wd4512 /wd4310 -Wundef -Wmissing-declarations -Wuninitialized -Wshadow -Wdeprecated-declarations -Wstrict-aliasing -Wtautological-compare)

ocv_define_module(musalegacy opencv_core opencv_video
  OPTIONAL opencv_objdetect opencv_imgproc opencv_calib3d opencv_musaarithm opencv_musafilters opencv_musaimgproc)

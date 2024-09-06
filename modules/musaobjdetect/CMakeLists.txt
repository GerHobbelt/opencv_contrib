if(IOS OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
  ocv_module_disable(musaobjdetect)
endif()

set(the_description "MUSA-accelerated Object Detection")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow -Wstrict-aliasing)

ocv_define_module(musaobjdetect opencv_objdetect opencv_musaarithm opencv_musawarping OPTIONAL opencv_musalegacy WRAP python)

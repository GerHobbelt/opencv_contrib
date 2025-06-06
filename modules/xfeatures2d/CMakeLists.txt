set(the_description "Contributed/Experimental Algorithms for Salient 2D Features Detection")

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()
ocv_define_module(xfeatures2d opencv_core opencv_imgproc opencv_features2d opencv_3d OPTIONAL opencv_shape opencv_ml opencv_cudaarithm WRAP python java objc)

if(NOT OPENCV_SKIP_FEATURES2D_DOWNLOADING)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_vgg.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_boostdesc.cmake)
  set(DOWNLOAD_DIR "${OpenCV_BINARY_DIR}/downloads/xfeatures2d")
  download_boost_descriptors("${DOWNLOAD_DIR}" boost_status)
  download_vgg_descriptors("${DOWNLOAD_DIR}" vgg_status)
  if(boost_status)
    ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/boostdesc.cpp "OPENCV_XFEATURES2D_HAS_BOOST_DATA=1")
  else()
    message(WARNING "features2d: Boost descriptor implementation is not available due to missing data (download failed: https://github.com/opencv/opencv_contrib/issues/1301)")
  endif()
  if(vgg_status)
    ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/vgg.cpp "OPENCV_XFEATURES2D_HAS_VGG_DATA=1")
  else()
    message(WARNING "features2d: VGG descriptor implementation is not available due to missing data (download failed: https://github.com/opencv/opencv_contrib/issues/1301)")
  endif()

  if(boost_status OR vgg_status)
    ocv_module_include_directories("${DOWNLOAD_DIR}")
  endif()
endif()

if(TARGET opencv_test_${name})
  ocv_target_include_directories(opencv_test_${name} "${OpenCV_SOURCE_DIR}/modules")  # use common files from features2d tests
  if(boost_status)
    ocv_target_compile_definitions(opencv_test_${name} PRIVATE "OPENCV_XFEATURES2D_HAS_BOOST_DATA=1")
  endif()
  if(vgg_status)
    ocv_target_compile_definitions(opencv_test_${name} PRIVATE "OPENCV_XFEATURES2D_HAS_VGG_DATA=1")
  endif()
endif()

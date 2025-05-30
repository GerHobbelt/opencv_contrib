set(BUILD_opencv_cnn_3dobj_INIT OFF)  # Must be disabled by default - requires custom build of Caffe.
if(NOT BUILD_opencv_cnn_3dobj)
  return()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Caffe)
if(Caffe_FOUND)
  message(STATUS "Caffe:   YES")
  set(HAVE_CAFFE 1)
else()
  message(STATUS "Caffe:   NO")
endif()

if(NOT BUILD_PROTOBUF)
  find_package(Protobuf)
endif()
if(Protobuf_FOUND)
  message(STATUS "Protobuf:   YES")
  set(HAVE_PROTOBUF 1)
else()
  message(STATUS "Protobuf:   NO")
endif()

find_package(Glog)
if(Glog_FOUND)
  message(STATUS "Glog:   YES")
  set(HAVE_GLOG 1)
else()
  message(STATUS "Glog:   NO")
endif()

if(NOT HAVE_CAFFE)
    ocv_module_disable(cnn_3dobj)
else()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${Caffe_INCLUDE_DIR})
set(the_description "CNN for 3D object recognition and pose estimation including a completed Sphere View on 3D objects")
ocv_define_module(cnn_3dobj opencv_core opencv_imgproc ${Caffe_LIBS} ${Glog_LIBS} ${Protobuf_LIBS} OPTIONAL opencv_features2d opencv_viz opencv_3d WRAP python)
ocv_add_testdata(testdata/cv contrib/cnn_3dobj)

if(TARGET opencv_test_cnn_3dobj)
  target_link_libraries(opencv_test_cnn_3dobj PUBLIC boost_system)
endif()

foreach(exe_TGT  classify video sphereview_data model_analysis)
    if(TARGET example_cnn_3dobj_${exe_TGT})
      target_link_libraries(example_cnn_3dobj_${exe_TGT} PUBLIC boost_system)
    endif()
endforeach()
endif()

set(test_deps opencv_mudev opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui opencv_ts ${OPENCV_MODULE_opencv_ts_DEPS})

ocv_check_dependencies(${test_deps})

if(OCV_DEPENDENCIES_FOUND)
  set(the_target "opencv_test_${name}")

  ocv_module_include_directories("${test_deps}" "${the_module}")

  file(GLOB test_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.mu")
  file(GLOB test_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
  source_group("Src" FILES ${test_srcs})
  source_group("Include" FILES ${test_hdrs})
  set(OPENCV_TEST_${the_module}_SOURCES ${test_srcs} ${test_hdrs})

  # ocv_musa_filter_options()

  # if(MUSA_VERSION VERSION_LESS "2.0")
  #   ocv_update(OPENCV_MUSA_OPTIONS_opencv_test_mudev "-std=c++11")
  # else()
  #   ocv_update(OPENCV_MUSA_OPTIONS_opencv_test_mudev "-std=c++14")
  #   ocv_warnings_disable(CMAKE_CXX_FLAGS -Wdeprecated-declarations)
  # endif()

  # MUSA_ADD_EXECUTABLE(${the_target} ${OPENCV_TEST_${the_module}_SOURCES} OPTIONS ${OPENCV_MUSA_OPTIONS_opencv_test_mudev})
  # TODO: extract to a separate file
  set(musa_src)
  set(cpp_src)
  foreach(file ${OPENCV_TEST_${the_module}_SOURCES})
    if(${file} MATCHES ".*\\.mu$")
      list(APPEND musa_src ${file})
    elseif(${file} MATCHES ".*\\.cpp$")
      list(APPEND cpp_src ${file})
    endif()
  endforeach()
  set_source_files_properties(${musa_src}
      PROPERTIES
      LANGUAGE CXX)
  add_library(${the_target}_musa_obj OBJECT ${musa_src})
  target_compile_options(${the_target}_musa_obj PRIVATE "--offload-arch=mp_21")
  target_compile_options(${the_target}_musa_obj PRIVATE -Wno-deprecated-declarations -Wno-shift-count-overflow)

  add_executable(${the_target} ${cpp_src} $<TARGET_OBJECTS:${the_target}_musa_obj>)
  ocv_target_link_libraries(${the_target} PRIVATE
      ${test_deps} ${OPENCV_LINKER_LIBS} ${MUSA_LIBRARIES}
  )
  add_dependencies(opencv_tests ${the_target})

  set_target_properties(${the_target} PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL}")
  set_source_files_properties(${OPENCV_TEST_${the_module}_SOURCES} ${${the_target}_pch}
    PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};AccuracyTest")

  # Additional target properties
  set_target_properties(${the_target} PROPERTIES
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
  )

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_target} PROPERTIES FOLDER "tests accuracy")
  endif()

  ocv_add_test_from_target("${the_target}" "Accuracy" "${the_target}")

  if(INSTALL_TESTS)
    install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_TEST_INSTALL_PATH} COMPONENT tests)
  endif()
endif()

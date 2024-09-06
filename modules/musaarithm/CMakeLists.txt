if(IOS OR WINRT OR (NOT HAVE_MUSA AND NOT BUILD_MUSA_STUBS))
  ocv_module_disable(musaarithm)
endif()

set(the_description "MUSA-accelerated Operations on Matrices")

ocv_add_module(musaarithm opencv_core OPTIONAL opencv_mudev WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow -Wunused-function -Wsign-compare -Wunused-parameter -Wmissing-prototypes -Wdeprecated-declarations -Wdeprecated-copy -Wabsolute-value -Wmacro-redefined)

# warning disable, may be deleted in future
# ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-function)


ocv_module_include_directories()
ocv_glob_module_sources()

set(extra_libs "")

if(HAVE_MUBLAS)
  list(APPEND extra_libs ${MUSA_mublas_LIBRARY})
endif()

if(HAVE_MUFFT)
  list(APPEND extra_libs ${MUSA_mufft_LIBRARY})
endif()

ocv_create_module(${extra_libs})

ocv_add_accuracy_tests(DEPENDS_ON opencv_imgproc)
ocv_add_perf_tests(DEPENDS_ON opencv_imgproc)

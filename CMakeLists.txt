cmake_minimum_required(VERSION 3.12)

project(mppi_generic_vendor LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  set(SKIP_BUILD TRUE)
else()
  set(SKIP_BUILD FALSE)
endif()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mppi_generic_vendor-extras.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mppi_generic_vendor-extras.cmake"
    @ONLY
)
if(SKIP_BUILD)
    message(WARNING "Skipping build of mppi_generic_vendor because CUDA is not available")
    ament_package(
      CONFIG_EXTRAS "${CMAKE_CURRENT_BINARY_DIR}/mppi_generic_vendor-extras.cmake"
    )
    return()
endif()

find_package(ament_cmake_vendor_package REQUIRED)

# Set the location where the external project will be installed
set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/mppi_generic_install)

ament_vendor(mppi_generic
  VCS_URL https://github.com/acdslab/mppi-generic.git
  VCS_VERSION v0.9.0
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DMPPI_BUILD_TESTS=OFF    # Turn off tests to speed up workspace builds
    -DMPPI_BUILD_EXAMPLES=OFF # Turn off examples
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    ${FORWARD_CUDA_ARGS}
)

# MPPI-Generic creates an MPPI-GenericConfig.cmake file so we export it
ament_export_dependencies(MPPI-Generic)

ament_package(
  CONFIG_EXTRAS "${CMAKE_CURRENT_BINARY_DIR}/mppi_generic_vendor-extras.cmake"
)
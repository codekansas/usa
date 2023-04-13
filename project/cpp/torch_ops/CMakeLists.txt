set(LIBRARY_NAME torch_ops)

# Gets all source files.
file(GLOB_RECURSE SRC_FILES *.c *.cpp)

# Finds PyTorch and Python libraries.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# Links relevant directories.
link_directories(${TORCH_INSTALL_PREFIX}/lib)
if(DEFINED ENV{CONDA_PREFIX})
  link_directories($ENV{CONDA_PREFIX}/lib)
endif()

add_library(${LIBRARY_NAME} SHARED ${SRC_FILES})

# Sets shared library properties.
set_target_properties(${LIBRARY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                 ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${LIBRARY_NAME} PROPERTIES PREFIX "")
set_target_properties(${LIBRARY_NAME} PROPERTIES OUTPUT_NAME ${LIBRARY_NAME})

# Links with Torch directories.
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC ${TORCH_INCLUDE_DIRS}
  PUBLIC ${PYTHON_INCLUDE_DIRS}
  PUBLIC ${pybind11_INCLUDE_DIRS})
target_link_libraries(
  ${LIBRARY_NAME}
  PUBLIC ${TORCH_LIBRARIES}
  PUBLIC ${TORCH_PYTHON_LIBRARY}
  PUBLIC ${pybind11_LIBRARIES})

cmake_minimum_required(VERSION 3.5)

project(linear LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "C:/software/libtorch/debug-1.6/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
find_package(Boost)

add_executable(linear
  main.cxx
  CSVReader.h
  CSVReader.cxx
  Net.h
  Train.h
#  Train.cxx
  Test.h
#  Test.cxx
  Data.h
#  Data.cxx
  IO.h
#  IO.cxx
    Config.h
    Config.cxx
)
target_link_libraries(linear "${TORCH_LIBRARIES}")
set_property(TARGET linear PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET linear
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:linear>)
endif (MSVC)
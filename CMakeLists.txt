cmake_minimum_required(VERSION 3.15)
project(erg_python C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Python
set(Python3_ROOT_DIR "C:/Users/ipg-pc-59/AppData/Roaming/uv/python/cpython-3.13.9-windows-x86_64-none")
find_package(Python3 REQUIRED COMPONENTS Development)

# NumPy include directory
set(NUMPY_INCLUDE_DIR "C:/workspace/erg-python/venv/Lib/site-packages/numpy/_core/include")

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

# Add the Python extension module
Python3_add_library(erg_python MODULE
    src/erg_module.c
)

# Link against the static library
target_link_libraries(erg_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/liberg.a
)

# Set compiler flags
if(MSVC)
    target_compile_options(erg_python PRIVATE /W4)
else()
    target_compile_options(erg_python PRIVATE -Wall -Wextra -mavx2)
endif()

# Set output directory
set_target_properties(erg_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/erg_python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/erg_python
)

# Remove the 'lib' prefix on Windows if present
set_target_properties(erg_python PROPERTIES PREFIX "")

# Ensure .pyd extension on Windows
if(WIN32)
    set_target_properties(erg_python PROPERTIES SUFFIX ".pyd")
endif()

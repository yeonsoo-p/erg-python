cmake_minimum_required(VERSION 3.15)
project(erg_python C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Development NumPy)
find_package(OpenMP REQUIRED)
# NumPy include directory
set(NUMPY_INCLUDE_DIR "${Python3_NumPy_INCLUDE_DIRS}")

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

# Link against the static library and required C++ runtime
target_link_libraries(erg_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/liberg.a
    -static-libgcc
    -Wl,-Bstatic
    -lstdc++
    -lgomp
    -lwinpthread
    -ldl
    -Wl,-Bdynamic
)

# Set compiler flags
target_compile_options(erg_python PRIVATE -Wall -Wextra -mavx2 ${OpenMP_C_FLAGS})

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

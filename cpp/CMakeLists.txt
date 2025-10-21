cmake_minimum_required(VERSION 3.10)
project(PiFlightController VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Controller library
add_library(piflight_controller STATIC
    PiFlightController.cpp
    PiFlightController.h
)

target_include_directories(piflight_controller PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Example executable
add_executable(piflight_example
    example.cpp
)

target_link_libraries(piflight_example PRIVATE
    piflight_controller
)

# Install targets
install(TARGETS piflight_controller
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES PiFlightController.h
    DESTINATION include/piflight
)

# Print build info
message(STATUS "π-Flight Controller Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")

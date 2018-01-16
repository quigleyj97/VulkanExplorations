cmake_minimum_required(VERSION 3.8)
project(VulkanExplorations)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan)

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Couldn't find vulkan!")
else()
    message(STATUS ${Vulkan_LIBRARIES})
    message(STATUS ${Vulkan_INCLUDE_DIRS})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_DOCS OFF)
add_subdirectory(external/glfw)

add_definitions(-DGLM_FORCE_RADIANS)
add_definitions(-DGLM_FORCE_DEPTH_ZERO_TO_ONE)
include_directories(external/gli external/glm external/glfw/include ${Vulkan_INCLUDE_DIRS})

set(BUILD_SHARED_LIBS OFF)

file(GLOB SHADERS shaders/*.spv)
file(GLOB SHADERSRC shaders/*.frag shaders/*.vert)
source_group("Shaders" FILES ${SHADERSRC})

file(GLOB RESOURCES res/*)
source_group("Resources" FILES ${RESOURCES})

file(COPY ${SHADERS} DESTINATION "${CMAKE_BINARY_DIR}/bin/shaders")
file(COPY ${RESOURCES} DESTINATION "${CMAKE_BINARY_DIR}/bin/res")

add_executable(VulkanExplorations HelloVulkan.cpp ${outres})
target_link_libraries(VulkanExplorations ${Vulkan_LIBRARIES} glfw ${WINLIBS})
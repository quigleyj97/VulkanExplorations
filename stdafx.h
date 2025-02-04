// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#define _CRT_SECURE_NO_WARNINGS 1

// std
#include <iostream>
#include <stdexcept>
#include <functional>
#include <vector>
#include <array>
#include <algorithm>
#include <set>
#include <fstream>
#include <iomanip>
#include <chrono>

// 3rd party
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

// rendering library
#include <vulkan/vulkan.hpp>

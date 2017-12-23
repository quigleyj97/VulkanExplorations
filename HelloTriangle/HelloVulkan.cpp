// HelloVulkan.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int main() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(640, 480, "Hello Vulkan!", nullptr, nullptr);

	uint32_t nExtensions = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &nExtensions, nullptr);

	std::cout << nExtensions << " extensions supported" << std::endl;

	glm::mat4 matrix;
	glm::vec4 vec;
	auto test = matrix * vec;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	glfwTerminate();

	return 0;
}


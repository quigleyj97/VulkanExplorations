#include "stdafx.h"

std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

class ApplicationModule {
public:
	void run() {
		initWindow();
		initVulkan();
		main();
		destroy();
	}
	const int WindowWidth = 800;
	const int WindowHeight = 600;
private:
	GLFWwindow * window = nullptr;
	VkInstance instance = nullptr;

	#pragma region
	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WindowWidth, WindowHeight, "HelloVulkan", nullptr, nullptr);
	}

	void initVulkan() {
		createInstance();
	}

	void main() const {
		while(!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void destroy() const {
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}
	#pragma endregion Lifecycle functions

	#pragma region 
	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("Not all requested validation layers were found");
		}
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
		appInfo.pEngineName = "BeepBoop";
		appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions = checkExtensionSupport(glfwExtensionCount);

		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		const VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

		if (result != VK_SUCCESS) {
			throw std::runtime_error("Initialization failed at Vulkan instance creation");
		}
	}

	/**
	 * \brief enumerates supported extensions and logs if there aren't enough for GLFW to start a window.
	 * \param glfwExtensionCount [out] Number of extensions GLFW needs
	 * \return returns GLFW required extensions
	 */
	const char** checkExtensionSupport(uint32_t &glfwExtensionCount) const {
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

		const auto glfwExtensionContainer = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<std::string> glfwExtensions;
		for (uint32_t i = 0; i < glfwExtensionCount; i++) {
			const std::string extensionName = glfwExtensionContainer[i];
			glfwExtensions.push_back(extensionName);
		}

		std::cout << "-----------------" << std::endl;
		std::cout << "Extensions supported: " << extensionCount << std::endl;
		std::cout << "Extensions required: " << glfwExtensionCount << std::endl;

		std::cout << "\tName\tVersion" << std::endl;

		std::vector<std::string> vulkanExtensions;
		for (const auto extension : extensions) {
			const std::string extensionName = extension.extensionName;
			vulkanExtensions.push_back(extensionName);
			std::cout << "\t" << extension.extensionName << "\t" << extension.specVersion << std::endl;
		}

		std::sort(vulkanExtensions.begin(), vulkanExtensions.end());
		std::sort(glfwExtensions.begin(), glfwExtensions.end());

		const bool extensionsSupported = std::includes(vulkanExtensions.begin(), vulkanExtensions.end(),
			glfwExtensions.begin(), glfwExtensions.end());

		std::cout << "Required extensions included: " << extensionsSupported << std::endl;

		return glfwExtensionContainer;
	}

	bool checkValidationLayerSupport() const {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		std::cout << "Avaliable Instance Layers" << std::endl;
		std::cout << "\tLayer Name\tDescription" << std::endl;

		for (const auto& layerProperties : availableLayers) {
			std::cout << "\t" << layerProperties.layerName << "\t" << layerProperties.description << std::endl;
		}

		for (const char* layerName: validationLayers) {
			bool layerFound = false;

			for(const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				std::cout << "Layer not found: " << layerName << std::endl;
				return false;
			}
		}

		return true;
	}
	#pragma endregion Vulkan utility functions
};

int main() {
	ApplicationModule app;

	try {
		app.run();
	} catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}


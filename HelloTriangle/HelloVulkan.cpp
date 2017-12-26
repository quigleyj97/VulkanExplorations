#include "stdafx.h"
#include <iomanip>

std::vector<std::string> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugReportCallbackEXT(vk::Instance instance, 
									  const VkDebugReportCallbackCreateInfoEXT* createInfo, 
									  const VkAllocationCallbacks* allocator, 
									  VkDebugReportCallbackEXT* callback) {
	const auto func = PFN_vkCreateDebugReportCallbackEXT(instance.getProcAddr("vkCreateDebugReportCallbackEXT"));

	if (func != nullptr) {
		return func(instance, createInfo, allocator, callback);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugReportCallbackEXT(vk::Instance instance,
								   const VkDebugReportCallbackEXT callback,
								   const VkAllocationCallbacks* allocator) {
	const auto func = PFN_vkDestroyDebugReportCallbackEXT(instance.getProcAddr("vkDestroyDebugReportCallbackEXT"));
	if (func != nullptr) {
		func(instance, callback, allocator);
	}
}

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
	vk::Instance instance;
	vk::DebugReportCallbackEXT callback;
	vk::PhysicalDevice physicalDevice = nullptr;

	#pragma region
	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WindowWidth, WindowHeight, "HelloVulkan", nullptr, nullptr);
	}

	void initVulkan() {
		createInstance();
		setupVulkanCallback();
		pickPhysicalDevice();
	}

	void main() const {
		while(!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void destroy() const {
		DestroyDebugReportCallbackEXT(instance, callback, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}
	#pragma endregion Lifecycle functions

	#pragma region 
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		vk::DebugReportFlagsEXT flags,
		vk::DebugReportObjectTypeEXT objType,
		uint64_t obj,
		size_t location,
		int32_t code,
		const char* layerPrefix,
		const char* msg,
		void* userData) {
		std::cerr << "Validation layer: " << msg << std::endl;

		return VK_FALSE;
	}

	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("Not all requested validation layers were found");
		}

		uint32_t glfwExtensionCount = 0;
		char** glfwExtensions = checkExtensionSupport(glfwExtensionCount);

		vk::ApplicationInfo appInfo = vk::ApplicationInfo()
			.setPApplicationName("Hello Triangle")
			.setApplicationVersion(VK_MAKE_VERSION(0, 1, 0))
			.setPEngineName("BeepBoop")
			.setEngineVersion(VK_MAKE_VERSION(0, 1, 0))
			.setApiVersion(VK_API_VERSION_1_0);

		vk::InstanceCreateInfo createInfo = vk::InstanceCreateInfo()
			.setPApplicationInfo(&appInfo)
			.setEnabledExtensionCount(glfwExtensionCount)
			.setPpEnabledExtensionNames(glfwExtensions);

		if (enableValidationLayers) {
			char** validationLayers_corrected = new char*[validationLayers.size()];
			for (int i = 0; i < validationLayers.size(); i++) {
				char *pc = new char[validationLayers[i].size() + 1];
				std::strcpy(pc, validationLayers[i].c_str());
				validationLayers_corrected[i] = pc;
			}
			createInfo.setEnabledLayerCount(static_cast<uint32_t>(validationLayers.size()));
			createInfo.setPpEnabledLayerNames(validationLayers_corrected);
		} else {
			createInfo.setEnabledLayerCount(0);
		}

		const vk::Result result = vk::createInstance(&createInfo, nullptr, &instance);

		if (result != vk::Result::eSuccess) {
			throw std::runtime_error("Initialization failed at Vulkan instance creation");
		}
	}

	std::vector<std::string> getRequiredExtensions() const {
		std::vector<std::string> extensions;

		uint32_t glfwExtensionCount;
		const char** glfwExtensionContainer = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
			const std::string extensionName = glfwExtensionContainer[i];
			extensions.push_back(extensionName);
		}

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		return extensions;
	}

	/**
	 * \brief enumerates supported extensions and logs if there aren't enough for GLFW to start a window.
	 * \param requiredExtensionCount [out] Number of extensions that the app needs
	 * \return returns array of required extensions
	 */
	char** checkExtensionSupport(uint32_t &requiredExtensionCount) const {
		std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

		std::vector<std::string> requiredExtensions = getRequiredExtensions();

		std::cout << "--EXTENSION REPORT--" << std::endl;
		std::cout << "Extensions supported: " << extensions.size() << std::endl;
		std::cout << "Extensions required: " << requiredExtensions.size() << std::endl;

		std::cout << "\t" << std::left << std::setw(40) << std::setfill(' ') << "Name";
		std::cout << std::left << std::setw(25) << std::setfill(' ') << "Version" << std::endl;

		std::vector<std::string> vulkanExtensions;
		for (const auto extension : extensions) {
			const std::string extensionName = extension.extensionName;
			vulkanExtensions.push_back(extensionName);
			std::cout << "\t" << std::left << std::setw(40) << std::setfill(' ') << extension.extensionName;
			std::cout << std::left << std::setw(25) << std::setfill(' ') << extension.specVersion << std::endl;
		}

		std::sort(vulkanExtensions.begin(), vulkanExtensions.end());
		std::sort(requiredExtensions.begin(), requiredExtensions.end());

		const bool extensionsSupported = std::includes(vulkanExtensions.begin(), vulkanExtensions.end(),
			requiredExtensions.begin(), requiredExtensions.end());

		std::cout << "Required extensions included: " << (extensionsSupported ? "TRUE" : "FALSE") << std::endl;
		if (!extensionsSupported) {
			throw std::runtime_error("Not all required extensions were found in this environment");
		}

		requiredExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
		char** requestedExtensions = new char*[requiredExtensions.size()];
		for (int i = 0; i < requiredExtensions.size(); i++) {
			char *pc = new char[requiredExtensions[i].size() + 1];
			std::strcpy(pc, requiredExtensions[i].c_str());
			requestedExtensions[i] = pc;
		}

		return requestedExtensions;
	}

	bool checkValidationLayerSupport() const {		
		std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

		std::cout << "--INSTANCE LAYER REPORT--" << std::endl;
		std::cout << "Avaliable Instance Layers" << std::endl;
		std::cout << "\t" << std::left << std::setw(40) << std::setfill(' ') << "Layer Name";
		std::cout << "Description" << std::endl;

		for (const auto layerProperties : availableLayers) {
			std::cout << "\t" << std::left << std::setw(40) << std::setfill(' ') << layerProperties.layerName;
			std::cout << layerProperties.description << std::endl;
		}

		for (const std::string validationLayerName: validationLayers) {
			bool layerFound = false;

			for (const vk::LayerProperties layer : availableLayers) {
				if (layer.layerName == validationLayerName) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				std::cout << "Layer not found: " << validationLayerName << std::endl;
				return false;
			}
		}

		return true;
	}

	void setupVulkanCallback() {
		if (!enableValidationLayers) return;

		const vk::DebugReportCallbackCreateInfoEXT createInfo = vk::DebugReportCallbackCreateInfoEXT()
			.setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning)
		// we have to do this because vulkan.hpp doesn't really have a type-safe way of setting the callback
			.setPfnCallback(PFN_vkDebugReportCallbackEXT(PFN_vkDebugReportCallbackEXT(debugCallback)));

		const VkDebugReportCallbackCreateInfoEXT createInfoCopy = createInfo;
		VkDebugReportCallbackEXT callbackCopy;
		
		if (CreateDebugReportCallbackEXT(instance, &createInfoCopy, nullptr, &callbackCopy) != VK_SUCCESS) {
			throw std::runtime_error("Error in initializing validation layer debug callback");
		}
		callback = vk::DebugReportCallbackEXT(callbackCopy);
	}

	void pickPhysicalDevice() {
		std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

		if (devices.size() == 0) {
			throw std::runtime_error("Couldn't find a GPU with Vulkan support! Check your drivers");
		}

		for (const auto& device: devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == vk::PhysicalDevice(nullptr)) {
			throw std::runtime_error("Couldn't find a GPU suitable for use!");
		}
	}

	bool isDeviceSuitable(vk::PhysicalDevice device) const {
		vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
		vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();

		// TODO: do some more advanced scoring in the future
		return deviceFeatures.geometryShader;
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

	std::cout << "Press any key to continue...";
	std::cin.get();

	return EXIT_SUCCESS;
}


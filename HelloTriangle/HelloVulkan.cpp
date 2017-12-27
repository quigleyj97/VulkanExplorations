#include "stdafx.h"
#include <iomanip>

const std::vector<std::string> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
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

struct QueueFamilyIndices {
	int graphicsFamily = -1;
	int presentFamily = -1;

	bool isComplete() {
		return graphicsFamily >= 0 && presentFamily >= 0;
	}
};

struct SwapChainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

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
	vk::Device device;
	vk::Queue graphicsQueue;
	vk::SurfaceKHR surface;
	vk::Queue presentationQueue;
	vk::SwapchainKHR swapChain;
	std::vector<vk::Image> swapChainImages;
	vk::Format swapChainImageFormat = vk::Format::eUndefined;
	vk::Extent2D swapChainExtent;
	std::vector<vk::ImageView> swapChainImageViews;

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
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
	}

	void main() const {
		while(!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void destroy() const {
		for (const auto imageView : swapChainImageViews) {
			device.destroyImageView(imageView);
		}
		device.destroySwapchainKHR(swapChain);
		device.destroy(nullptr);
		DestroyDebugReportCallbackEXT(instance, callback, nullptr);
		instance.destroySurfaceKHR(surface, nullptr);
		instance.destroy(nullptr);

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
		
		#pragma region
		std::cout << "--DEVICE REPORT--" << std::endl;
		std::cout << "\t" << std::left << std::setw(20) << std::setfill(' ') << "Property";
		std::cout << "Value" << std::endl;
		
		std::cout << "\t" << std::left << std::setw(20) << std::setfill(' ') << "Name";
		std::cout << deviceProperties.deviceName << std::endl;

		std::cout << "\t" << std::left << std::setw(20) << std::setfill(' ') << "Type";
		std::cout << vk::to_string(deviceProperties.deviceType) << std::endl;

		std::cout << "\t" << std::left << std::setw(20) << std::setfill(' ') << "Vendor";
		std::cout << deviceProperties.vendorID << std::endl;

		std::cout << "\t" << std::left << std::setw(20) << std::setfill(' ') << "API Version";
		std::cout << deviceProperties.apiVersion << std::endl;		
		#pragma endregion Device Report

		QueueFamilyIndices indices = findQueueFamilies(device);

		const bool extensionsSupported = checkDeviceExtensionSupport(device);

		std::cout << "Device supports required extensions: ";
		std::cout << (extensionsSupported ? "TRUE" : "FALSE") << std::endl;

		if (!extensionsSupported) {
			return false;
		}

		bool swapChainsSupported = false;
		SwapChainSupportDetails supportDetails;
		if (extensionsSupported) {
			supportDetails = querySwapChainSupport(device);
			swapChainsSupported = !supportDetails.formats.empty() && !supportDetails.presentModes.empty();
		}

#pragma region
		std::cout << "---SWAP CHAIN REPORT--" << std::endl;

		std::cout << "Swap chains supported: ";
		std::cout << (swapChainsSupported ? "TRUE" : "FALSE") << std::endl;

		if(!swapChainsSupported) {
			return false;
		}

		std::cout << "----FORMAT REPORT--" << std::endl;
		for (const auto& format : supportDetails.formats) {
			std::cout << "\tFormat: " << to_string(format.format) << std::endl;
			std::cout << "\tColor space: " << to_string(format.colorSpace) << std::endl;
			std::cout << "\t----" << std::endl;
		}

		std::cout << "----PRESENTATION MODE REPORT--" << std::endl;
		for (const auto& presentationMode : supportDetails.presentModes) {
			std::cout << "\t" << to_string(presentationMode) << std::endl;
		}
#pragma endregion Swap Chain report

		// TODO: do some more advanced scoring in the future
		return deviceFeatures.geometryShader
			&& indices.isComplete();
	}

	bool checkDeviceExtensionSupport(vk::PhysicalDevice device) const {
		const std::vector<vk::ExtensionProperties> availableExtensions = 
			device.enumerateDeviceExtensionProperties();

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) const {
		QueueFamilyIndices indices;

		std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

		int i = 0;
		for (const auto& queueFamily: queueFamilies) {
			const auto presentationSupport = device.getSurfaceSupportKHR(i, surface);
			if (queueFamily.queueCount > 0 
				&& queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
				indices.graphicsFamily = i;
			}

			if (queueFamily.queueCount > 0 && presentationSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			++i;
		}

		return indices;
	}

	void createLogicalDevice() {
		const QueueFamilyIndices indicies = findQueueFamilies(physicalDevice);

		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		std::set<int> uniqueQueueFamilies = { indicies.graphicsFamily, indicies.presentFamily };

		float priority = 1.0f;

		for (int queueFamily : uniqueQueueFamilies) {
			vk::DeviceQueueCreateInfo queueCreateInfo = vk::DeviceQueueCreateInfo()
				.setQueueFamilyIndex(queueFamily)
				.setQueueCount(1)
				.setPQueuePriorities(&priority);
			queueCreateInfos.push_back(queueCreateInfo);
		}
		
		vk::PhysicalDeviceFeatures deviceFeatures;

		vk::DeviceCreateInfo deviceCreateInfo = vk::DeviceCreateInfo()
			.setPQueueCreateInfos(queueCreateInfos.data())
			.setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()))
			.setPEnabledFeatures(&deviceFeatures)
			.setEnabledExtensionCount(static_cast<uint32_t>(deviceExtensions.size()))
			.setPpEnabledExtensionNames(deviceExtensions.data());

		if (enableValidationLayers) {
			char** validationLayers_corrected = new char*[validationLayers.size()];
			for (int i = 0; i < validationLayers.size(); i++) {
				char *pc = new char[validationLayers[i].size() + 1];
				std::strcpy(pc, validationLayers[i].c_str());
				validationLayers_corrected[i] = pc;
			}
			deviceCreateInfo
				.setEnabledLayerCount(static_cast<uint32_t>(validationLayers.size()))
				.setPpEnabledLayerNames(validationLayers_corrected);
		}
		else {
			deviceCreateInfo.setEnabledLayerCount(0);
		}

		if (physicalDevice.createDevice(&deviceCreateInfo, nullptr, &device) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create a logical device handle!");
		}

		graphicsQueue = device.getQueue(indicies.graphicsFamily, 0);
		presentationQueue = device.getQueue(indicies.presentFamily, 0);
	}

	void createSurface() {
		VkSurfaceKHR temp_surface;
		if (glfwCreateWindowSurface(instance, window, nullptr, &temp_surface) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create a window surface!");
		}

		surface = temp_surface;
	}

	SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device) const {
		SwapChainSupportDetails details;

		device.getSurfaceCapabilitiesKHR(surface, &details.capabilities);

		details.formats = device.getSurfaceFormatsKHR(surface);
		details.presentModes = device.getSurfacePresentModesKHR(surface);

		return details;
	}

	vk::SurfaceFormatKHR pickSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const {
		if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
			return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
		}

		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == vk::Format::eB8G8R8A8Unorm 
				&& availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
				return availableFormat;
			}
		}

		// TODO: Score remaining formats on how good they are and return the best
		std::cerr << "WARNING: Preferred swap surface format not found. Defaulting to first available" << std::endl;
		return availableFormats[0];
	}

	vk::PresentModeKHR pickSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const {
		vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;
		
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
				return availablePresentMode;
			} else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
				bestMode = availablePresentMode;
			}
		}

		return bestMode;
	}

	vk::Extent2D pickSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		} else {
			vk::Extent2D actualExtent = { static_cast<uint32_t>(WindowWidth), static_cast<uint32_t>(WindowHeight) };

			actualExtent.width = std::max(capabilities.minImageExtent.width,
				std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height,
				std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

	void createSwapChain() {
		const SwapChainSupportDetails supportDetails = querySwapChainSupport(physicalDevice);

		const vk::SurfaceFormatKHR surfaceFormat = pickSwapSurfaceFormat(supportDetails.formats);
		const vk::PresentModeKHR presentMode = pickSwapPresentMode(supportDetails.presentModes);
		const vk::Extent2D extent = pickSwapExtent(supportDetails.capabilities);

		uint32_t imageCount = supportDetails.capabilities.minImageCount + 1;
		if (supportDetails.capabilities.maxImageCount > 0 
			&& imageCount > supportDetails.capabilities.maxImageCount) {
			imageCount = supportDetails.capabilities.maxImageCount;
		}

		vk::SwapchainCreateInfoKHR createInfo = vk::SwapchainCreateInfoKHR()
			.setSurface(surface)
			.setMinImageCount(imageCount)
			.setImageFormat(surfaceFormat.format)
			.setImageColorSpace(surfaceFormat.colorSpace)
			.setImageExtent(extent)
			.setImageArrayLayers(1)
			.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);

		const QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		const std::array<uint32_t, 2> queueFamilyIndicies = { 
			static_cast<uint32_t>(indices.graphicsFamily), 
			static_cast<uint32_t>(indices.presentFamily)
		};

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo
				.setImageSharingMode(vk::SharingMode::eConcurrent)
				.setQueueFamilyIndexCount(2)
				.setPQueueFamilyIndices(queueFamilyIndicies.data());
		} else {
			createInfo
				.setImageSharingMode(vk::SharingMode::eExclusive);
		}

		createInfo
			.setPreTransform(supportDetails.capabilities.currentTransform)
			.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
			.setPresentMode(presentMode)
			.setClipped(VK_TRUE)
			.setOldSwapchain(nullptr);

		if (device.createSwapchainKHR(&createInfo, nullptr, &swapChain) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create swap chain");
		}

		swapChainImages = device.getSwapchainImagesKHR(swapChain);

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		std::transform(swapChainImageViews.begin(), swapChainImageViews.end(), swapChainImages.begin(), 
			swapChainImageViews.begin(),
			[&](const vk::ImageView& view, const vk::Image& input) {
			vk::ImageViewCreateInfo createInfo = vk::ImageViewCreateInfo()
				.setImage(input)
				.setViewType(vk::ImageViewType::e2D)
				.setFormat(swapChainImageFormat)
				// defaults are VK_COMPONENT_SWIZZLE_IDENTITY
				.setComponents(vk::ComponentMapping());

			vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
			
			createInfo.setSubresourceRange(range);

			vk::ImageView copy;
			if (device.createImageView(&createInfo, nullptr, &copy) != vk::Result::eSuccess) {
				throw std::runtime_error("Failed to create image vies in swap chain!");
			}
			return copy;
		});
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

	std::cout << "Press enter to quit...";
	std::cin.get();

	return EXIT_SUCCESS;
}


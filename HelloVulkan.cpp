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

// rendering library
#include <vulkan/vulkan.hpp>

// 3rd party
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gli/gli.hpp>
#include <gli/texture.hpp>

const std::array<std::string, 1> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};

const std::array<const char*, 1> deviceExtensions = {
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

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + filename);
	}

	size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();

	return buffer;
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

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription() {
		return vk::VertexInputBindingDescription()
			.setBinding(0)
			.setStride(sizeof(Vertex))
			.setInputRate(vk::VertexInputRate::eVertex);
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
		return std::array<vk::VertexInputAttributeDescription, 3> {
			vk::VertexInputAttributeDescription()
				.setBinding(0)
				.setLocation(0)
				.setFormat(vk::Format::eR32G32B32Sfloat)
				.setOffset(offsetof(Vertex, pos)),
			vk::VertexInputAttributeDescription()
				.setBinding(0)
				.setLocation(1)
				.setFormat(vk::Format::eR32G32B32Sfloat)
				.setOffset(offsetof(Vertex, color)),
			vk::VertexInputAttributeDescription()
				.setBinding(0)
				.setLocation(2)
				.setFormat(vk::Format::eR32G32Sfloat)
				.setOffset(offsetof(Vertex, texCoord))
		};
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

const std::array<Vertex, 8> vertices {
	Vertex {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
	Vertex {{ 0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	Vertex {{ 0.5f,  0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	Vertex {{-0.5f,  0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

	Vertex {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
	Vertex {{ 0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	Vertex {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	Vertex {{-0.5f,  0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
};


const std::array<uint16_t, 12> indices = {
	0, 1, 2, 2, 3, 0,
	4, 5, 6, 6, 7, 4
};

class ApplicationModule {
public:
	void run() {
		initWindow();
		initVulkan();
#ifndef NDEBUG
		printAllReports();
#endif
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
	vk::RenderPass renderPass;
	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::Pipeline graphicsPipeline;
	std::vector<vk::Framebuffer> swapChainFramebuffers;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;
	vk::Semaphore imageAvailableSemaphore;
	vk::Semaphore renderFinishedSemaphore;
	vk::Buffer vertexBuffer;
	vk::DeviceMemory vertexBufferMemory;
	vk::Buffer indexBuffer;
	vk::DeviceMemory indexBufferMemory;
	vk::Buffer uniformBuffer;
	vk::DeviceMemory uniformBufferMemory;
	std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::high_resolution_clock::now();
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSet descriptorSet;
	vk::Image textureImage;
	vk::DeviceMemory textureImageMemory;
	vk::ImageView textureImageView;
	vk::Format textureImageFormat = vk::Format::eUndefined;
	vk::Sampler textureSampler;

	#pragma region
	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WindowWidth, WindowHeight, "HelloVulkan", nullptr, nullptr);

		glfwSetWindowUserPointer(window, this);
		glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int width, int height) -> void {
			if (width == 0 || height == 0) return;
			auto app = reinterpret_cast<ApplicationModule*>(glfwGetWindowUserPointer(window));
			app->invalidateSwapChain();
		});
	}

	void initVulkan() {
		createInstance();
		setupVulkanCallback();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffer();
		createDescriptorPool();
		createDescriptorSet();
		createCommandBuffers();
		createSemaphores();
	}

	void main() {
		while(!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			updateUniformBuffer();
			drawFrame();
		}

		device.waitIdle();
	}

	void destroy() const {
		destroySwapChain();

		device.destroySampler(textureSampler);
		device.destroyImageView(textureImageView);
		device.destroyImage(textureImage);
		device.freeMemory(textureImageMemory);
		device.destroyDescriptorPool(descriptorPool);
		device.destroyDescriptorSetLayout(descriptorSetLayout);
		device.destroyBuffer(uniformBuffer);
		device.freeMemory(uniformBufferMemory);
		device.destroyBuffer(indexBuffer);
		device.freeMemory(indexBufferMemory);
		device.destroyBuffer(vertexBuffer);
		device.freeMemory(vertexBufferMemory);
		device.destroySemaphore(renderFinishedSemaphore);
		device.destroySemaphore(imageAvailableSemaphore);
		device.destroyCommandPool(commandPool);
		device.destroy(nullptr);
		DestroyDebugReportCallbackEXT(instance, callback, nullptr);
		instance.destroySurfaceKHR(surface, nullptr);
		instance.destroy(nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void updateUniformBuffer() {
		auto currentTime = std::chrono::high_resolution_clock::now();

		float t = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(1.0f), t * glm::half_pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::quarter_pi<float>(),
			static_cast<float>(swapChainExtent.width / swapChainExtent.height), 0.1f, 10.0f);

		ubo.proj[1][1] *= -1;

		// TODO: Change these to push constants
		void* data;
		device.mapMemory(uniformBufferMemory, 0, sizeof(ubo), {}, &data);
		std::memcpy(data, &ubo, sizeof(ubo));
		device.unmapMemory(uniformBufferMemory);
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
	 * \brief enumerates supported extensions and throws if there aren't enough for GLFW to start a window.
	 * \param requiredExtensionCount [out] Number of extensions that the app needs
	 * \return returns array of required extensions
	 * \throws std::runtime_error when required extensions couldn't be found
	 */
	char** checkExtensionSupport(uint32_t &requiredExtensionCount) const {
		std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

		std::vector<std::string> requiredExtensions = getRequiredExtensions();
		std::set<std::string> extensionSet(requiredExtensions.begin(), requiredExtensions.end());

		for (const auto extension : extensions) {
			extensionSet.erase(extension.extensionName);
		}

		if (extensionSet.size() > 0) {
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

		QueueFamilyIndices indices = findQueueFamilies(device);

		const bool extensionsSupported = checkDeviceExtensionSupport(device);

		if (!extensionsSupported) {
			return false;
		}

		bool swapChainsSupported = false;
		SwapChainSupportDetails supportDetails;
		if (extensionsSupported) {
			supportDetails = querySwapChainSupport(device);
			swapChainsSupported = !supportDetails.formats.empty() && !supportDetails.presentModes.empty();
		}

		// TODO: do some more advanced scoring in the future
		return swapChainsSupported 
			&& deviceFeatures.geometryShader
			&& deviceFeatures.samplerAnisotropy
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
		deviceFeatures.setSamplerAnisotropy(true);

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
			int width, height;
			glfwGetWindowSize(window, &width, &height);
			vk::Extent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

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
			return createImageView(input, swapChainImageFormat);
		});
	}

	void createGraphicsPipeline() {
		const auto vertShaderSrc = readFile("shaders/vert.spv");
		const auto fragShaderSrc = readFile("shaders/frag.spv");

		vk::ShaderModule vertShaderModule = createShaderModule(vertShaderSrc);
		vk::ShaderModule fragShaderModule = createShaderModule(fragShaderSrc);

		vk::PipelineShaderStageCreateInfo vertShaderCreateInfo = vk::PipelineShaderStageCreateInfo()
			.setStage(vk::ShaderStageFlagBits::eVertex)
			.setModule(vertShaderModule)
			.setPName("main");
		vk::PipelineShaderStageCreateInfo fragShaderCreateInfo = vk::PipelineShaderStageCreateInfo()
			.setStage(vk::ShaderStageFlagBits::eFragment)
			.setModule(fragShaderModule)
			.setPName("main");

		const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
			vertShaderCreateInfo,
			fragShaderCreateInfo
		};

		const auto bindingDescription = Vertex::getBindingDescription();
		const auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo = vk::PipelineVertexInputStateCreateInfo()
			.setVertexBindingDescriptionCount(1)
			.setPVertexBindingDescriptions(&bindingDescription)
			.setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()))
			.setPVertexAttributeDescriptions(attributeDescriptions.data());

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
			.setTopology(vk::PrimitiveTopology::eTriangleList)
			.setPrimitiveRestartEnable(false);

		vk::Viewport viewport = vk::Viewport()
			.setX(0.0f)
			.setY(0.0f)
			.setWidth(static_cast<float>(swapChainExtent.width))
			.setHeight(static_cast<float>(swapChainExtent.height))
			.setMinDepth(0.0f)
			.setMaxDepth(1.0f);

		vk::Rect2D scissor = vk::Rect2D()
			.setOffset({ 0, 0 })
			.setExtent(swapChainExtent);

		vk::PipelineViewportStateCreateInfo viewportState = vk::PipelineViewportStateCreateInfo()
			.setViewportCount(1)
			.setPViewports(&viewport)
			.setScissorCount(1)
			.setPScissors(&scissor);

		vk::PipelineRasterizationStateCreateInfo rasterizerCreateInfo = vk::PipelineRasterizationStateCreateInfo()
			.setDepthClampEnable(false)
			.setRasterizerDiscardEnable(false)
			.setPolygonMode(vk::PolygonMode::eFill)
			.setLineWidth(1.0f)
			.setCullMode(vk::CullModeFlagBits::eBack)
			.setFrontFace(vk::FrontFace::eCounterClockwise)
			.setDepthBiasEnable(false);

		vk::PipelineMultisampleStateCreateInfo multisampleCreateInfo = vk::PipelineMultisampleStateCreateInfo()
			.setSampleShadingEnable(false)
			.setRasterizationSamples(vk::SampleCountFlagBits::e1);

		vk::PipelineColorBlendAttachmentState colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
			.setColorWriteMask(vk::ColorComponentFlagBits::eR
				| vk::ColorComponentFlagBits::eG
				| vk::ColorComponentFlagBits::eB
				| vk::ColorComponentFlagBits::eA)
			.setBlendEnable(false);

		vk::PipelineColorBlendStateCreateInfo colorBlendCreateInfo = vk::PipelineColorBlendStateCreateInfo()
			.setLogicOpEnable(false)
			.setAttachmentCount(1)
			.setPAttachments(&colorBlendAttachment);

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo()
			.setSetLayoutCount(1)
			.setPSetLayouts(&descriptorSetLayout);

		if (device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create the shader pipeline layout");
		}

		vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo()
			.setStageCount(static_cast<uint32_t>(shaderStages.size()))
			.setPStages(shaderStages.data())
			.setPVertexInputState(&vertexInputInfo)
			.setPInputAssemblyState(&inputAssembly)
			.setPViewportState(&viewportState)
			.setPRasterizationState(&rasterizerCreateInfo)
			.setPMultisampleState(&multisampleCreateInfo)
			.setPColorBlendState(&colorBlendCreateInfo)
			.setLayout(pipelineLayout)
			.setRenderPass(renderPass)
			.setSubpass(0);

		if (device.createGraphicsPipelines(nullptr, 1, &pipelineInfo, nullptr, &graphicsPipeline)
			!= vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create graphics pipeline");
		}

		device.destroyShaderModule(fragShaderModule);
		device.destroyShaderModule(vertShaderModule);
	}

	vk::ShaderModule createShaderModule(const std::vector<char>& code) const {
		vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo()
			.setCodeSize(code.size())
			.setPCode(reinterpret_cast<const uint32_t*>(code.data()));

		vk::ShaderModule shaderModule;
		if (device.createShaderModule(&createInfo, nullptr, &shaderModule) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to load shader into graphics device!");
		}

		return shaderModule;
	}

	void createRenderPass() {
		vk::SubpassDependency dependency = vk::SubpassDependency()
			.setSrcSubpass(VK_SUBPASS_EXTERNAL)
			.setDstSubpass(0)
			.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
			.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
			.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

		vk::AttachmentDescription colorAttachment = vk::AttachmentDescription()
			.setFormat(swapChainImageFormat)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
			.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
			.setInitialLayout(vk::ImageLayout::eUndefined)
			.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

		vk::AttachmentReference colorAttachmentRef = vk::AttachmentReference()
			.setAttachment(0)
			.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

		vk::SubpassDescription subpass = vk::SubpassDescription()
			.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
			.setColorAttachmentCount(1)
			.setPColorAttachments(&colorAttachmentRef);

		vk::RenderPassCreateInfo renderPassCreateInfo = vk::RenderPassCreateInfo()
			.setAttachmentCount(1)
			.setPAttachments(&colorAttachment)
			.setSubpassCount(1)
			.setPSubpasses(&subpass)
			.setDependencyCount(1)
			.setPDependencies(&dependency);

		if (device.createRenderPass(&renderPassCreateInfo, nullptr, &renderPass) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create Vulkan render pass");
		}
	}

	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		std::transform(swapChainFramebuffers.begin(), swapChainFramebuffers.end(), 
					   swapChainImageViews.begin(), swapChainFramebuffers.begin(),
			[&](const auto& _, const auto& swapChainImageView) {
			std::array<vk::ImageView, 1> attachments = {
				swapChainImageView
			};

			vk::FramebufferCreateInfo framebufferCreateInfo = vk::FramebufferCreateInfo()
				.setRenderPass(renderPass)
				.setAttachmentCount(static_cast<uint32_t>(attachments.size()))
				.setPAttachments(attachments.data())
				.setWidth(swapChainExtent.width)
				.setHeight(swapChainExtent.height)
				.setLayers(1);

			return device.createFramebuffer(framebufferCreateInfo);
		});
	}

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vk::CommandPoolCreateInfo poolCreateInfo = vk::CommandPoolCreateInfo()
			.setQueueFamilyIndex(queueFamilyIndices.graphicsFamily);

		commandPool = device.createCommandPool(poolCreateInfo);
	}

	void createCommandBuffers() {
		vk::CommandBufferAllocateInfo allocateInfo = vk::CommandBufferAllocateInfo()
			.setCommandPool(commandPool)
			.setLevel(vk::CommandBufferLevel::ePrimary)
			.setCommandBufferCount(static_cast<uint32_t>(swapChainFramebuffers.size()));

		commandBuffers = device.allocateCommandBuffers(allocateInfo);

		std::transform(commandBuffers.begin(), commandBuffers.end(), swapChainFramebuffers.begin(), 
					   commandBuffers.begin(),
			[&](const vk::CommandBuffer& buffer, const vk::Framebuffer& swapChainFramebuffer) {
			vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo()
				.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

			buffer.begin(beginInfo);

			vk::ClearValue clearColor = 
				{ vk::ClearColorValue { std::array<float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } } }; // black

			vk::RenderPassBeginInfo renderPassBeginInfo = vk::RenderPassBeginInfo()
				.setRenderPass(renderPass)
				.setFramebuffer(swapChainFramebuffer)
				.setRenderArea(vk::Rect2D({ 0, 0 }, swapChainExtent))
				.setClearValueCount(1)
				.setPClearValues(&clearColor);

			buffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
				buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

				std::array<vk::Buffer, 1> buffers = { vertexBuffer };
				std::array<vk::DeviceSize, 1> offsets = { 0 };
				buffer.bindVertexBuffers(0, static_cast<uint32_t>(buffers.size()), buffers.data(), offsets.data());
				buffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
				buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, { descriptorSet }, {});

				buffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
			buffer.endRenderPass();
			buffer.end();
			return buffer;
		});
	}

	void drawFrame() {
		const vk::ResultValue<uint32_t> imageResult = device.acquireNextImageKHR(swapChain, 
			std::numeric_limits<uint64_t>::max(), 
			imageAvailableSemaphore, 
			nullptr);

		if (imageResult.result == vk::Result::eErrorOutOfDateKHR
			|| imageResult.result == vk::Result::eSuboptimalKHR) {
			invalidateSwapChain();
			return;
		}
		else if (imageResult.result != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to get swap chain images after render!");
		}

		uint32_t imageIndex = imageResult.value;

		const std::array<vk::Semaphore, 1> waitSemaphores {
			imageAvailableSemaphore
		};

		const std::array<vk::Semaphore, 1> signalSemaphores{
			renderFinishedSemaphore
		};

		const std::array<vk::PipelineStageFlags, 1> waitStages {
			vk::PipelineStageFlagBits::eColorAttachmentOutput
		};

		const vk::SubmitInfo submitInfo = vk::SubmitInfo()
			.setWaitSemaphoreCount(static_cast<uint32_t>(waitSemaphores.size()))
			.setPWaitSemaphores(waitSemaphores.data())
			.setPWaitDstStageMask(waitStages.data())
			.setCommandBufferCount(static_cast<uint32_t>(commandBuffers.size()))
			.setPCommandBuffers(commandBuffers.data())
			.setSignalSemaphoreCount(static_cast<uint32_t>(signalSemaphores.size()))
			.setPSignalSemaphores(signalSemaphores.data());

		graphicsQueue.submit(1, &submitInfo, nullptr);

		const std::array<vk::SwapchainKHR, 1> swapChains{ swapChain };

		const vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR()
			.setWaitSemaphoreCount(static_cast<uint32_t>(signalSemaphores.size()))
			.setPWaitSemaphores(signalSemaphores.data())
			.setSwapchainCount(static_cast<uint32_t>(swapChains.size()))
			.setPSwapchains(swapChains.data())
			.setPImageIndices(&imageIndex);

		presentationQueue.presentKHR(presentInfo);

		presentationQueue.waitIdle();
	}

	void createSemaphores() {
		const vk::SemaphoreCreateInfo semaphoreCreateInfo = vk::SemaphoreCreateInfo();

		imageAvailableSemaphore = device.createSemaphore(semaphoreCreateInfo);
		renderFinishedSemaphore = device.createSemaphore(semaphoreCreateInfo);
	}

	void invalidateSwapChain() {
		device.waitIdle();

		destroySwapChain();

		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandBuffers();
	}

	void destroySwapChain() const {
		for (const auto& framebuffer : swapChainFramebuffers) {
			device.destroyFramebuffer(framebuffer);
		}
		device.destroyPipeline(graphicsPipeline);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyRenderPass(renderPass);
		for (const auto imageView : swapChainImageViews) {
			device.destroyImageView(imageView);
		}
		device.destroySwapchainKHR(swapChain);
	}

	void createVertexBuffer() {
		const vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
		auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
		std::memcpy(data, vertices.data(), bufferSize);
		device.unmapMemory(stagingBufferMemory);

		std::tie(vertexBuffer, vertexBufferMemory) = createBuffer(bufferSize, 
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);
	}

	void createIndexBuffer() {
		const vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
		auto[stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
		std::memcpy(data, indices.data(), bufferSize);
		device.unmapMemory(stagingBufferMemory);

		std::tie(indexBuffer, indexBufferMemory) = createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);
	}

	void copyBuffer(const vk::Buffer srcBuffer, const vk::Buffer dstBuffer, const vk::DeviceSize size) const {
		vk::CommandBuffer buffer = getSingleCommandBuffer();

		vk::BufferCopy copyRegion = vk::BufferCopy()
			.setSize(size);

		buffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleCommandBuffer(buffer);
	}

	std::tuple<vk::Buffer, vk::DeviceMemory> createBuffer(const vk::DeviceSize size, 
														  const vk::BufferUsageFlags usage,
														  const vk::MemoryPropertyFlags properties) const {
		const vk::BufferCreateInfo bufferCreateInfo = vk::BufferCreateInfo()
			.setSize(size)
			.setUsage(usage)
			.setSharingMode(vk::SharingMode::eExclusive);

		vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

		const auto memoryRequirements = device.getBufferMemoryRequirements(buffer);

		const vk::MemoryAllocateInfo allocateInfo = vk::MemoryAllocateInfo()
			.setAllocationSize(memoryRequirements.size)
			.setMemoryTypeIndex(findMemoryType(memoryRequirements.memoryTypeBits, properties));

		vk::DeviceMemory bufferMemory = device.allocateMemory(allocateInfo);
		device.bindBufferMemory(buffer, bufferMemory, 0);

		return std::make_tuple(buffer, bufferMemory);
	}

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
		auto memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if (typeFilter & (1 << i)
				&& (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("Failed to find a suitable memory type!");
	}

	void createDescriptorSetLayout() {
		vk::DescriptorSetLayoutBinding uboLayoutBinding = vk::DescriptorSetLayoutBinding()
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eVertex);

		vk::DescriptorSetLayoutBinding samplerLayoutBinding;
		samplerLayoutBinding.setBinding(1)
			.setDescriptorCount(1)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setPImmutableSamplers(nullptr)
			.setStageFlags(vk::ShaderStageFlagBits::eFragment);

		std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
			uboLayoutBinding,
			samplerLayoutBinding
		};

		vk::DescriptorSetLayoutCreateInfo layoutCreateInfo = vk::DescriptorSetLayoutCreateInfo()
			.setBindingCount(static_cast<uint32_t>(bindings.size()))
			.setPBindings(bindings.data());

		descriptorSetLayout = device.createDescriptorSetLayout(layoutCreateInfo);
	}

	void createUniformBuffer() {
		vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
		std::tie(uniformBuffer, uniformBufferMemory) = createBuffer(bufferSize, 
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	}

	void createDescriptorPool() {
		std::array<vk::DescriptorPoolSize, 2> poolSizes = {
			vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 1},
			vk::DescriptorPoolSize {vk::DescriptorType::eCombinedImageSampler, 1}
		};

		vk::DescriptorPoolCreateInfo poolCreateInfo = vk::DescriptorPoolCreateInfo()
			.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
			.setPPoolSizes(poolSizes.data())
			.setMaxSets(1);

		descriptorPool = device.createDescriptorPool(poolCreateInfo);
	}

	void createDescriptorSet() {
		std::array<vk::DescriptorSetLayout, 1> layouts = { descriptorSetLayout };
		vk::DescriptorSetAllocateInfo allocateInfo = vk::DescriptorSetAllocateInfo()
			.setDescriptorPool(descriptorPool)
			.setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
			.setPSetLayouts(layouts.data());

		auto tempBag = device.allocateDescriptorSets(allocateInfo);
		descriptorSet = tempBag[0];

		vk::DescriptorBufferInfo bufferInfo;
		bufferInfo.setBuffer(uniformBuffer)
			.setOffset(0)
			.setRange(sizeof(UniformBufferObject));

		vk::DescriptorImageInfo imageInfo(textureSampler, textureImageView, 
			vk::ImageLayout::eShaderReadOnlyOptimal);

		std::array<vk::WriteDescriptorSet, 2> writeDescriptorSets = {
			vk::WriteDescriptorSet {descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo},
			vk::WriteDescriptorSet {descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo}
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	// todo: generic-ize this function, make it not throw
	void createTextureImage() {
		auto texture = gli::load("./res/breadtangle.dds");
		if (texture.empty()) {
			throw std::runtime_error("Texture not found");
		}

		gli::texture2d texData(texture);
		textureImageFormat = vk::Format(texData.format());

		vk::DeviceSize imageSize = texData.size();

		auto[stagingBuffer, stagingBufferMemory] = createBuffer(imageSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		void *data;
		device.mapMemory(stagingBufferMemory, 0, imageSize, {}, &data);
		std::memcpy(data, texData.data(), static_cast<size_t>(imageSize));
		device.unmapMemory(stagingBufferMemory);

		std::tie(textureImage, textureImageMemory) = createImage(texData, textureImageFormat,
			vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal);

		transitionImageLayout(textureImage, textureImageFormat, 
			vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		copyBufferToImage(stagingBuffer, textureImage, texData.extent());

		transitionImageLayout(textureImage, textureImageFormat,
			vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);
	}

	std::tuple<vk::Image, vk::DeviceMemory> createImage(gli::texture2d texData, 
														vk::Format format, 
														vk::ImageTiling tiling, 
														vk::ImageUsageFlags usage, 
														vk::MemoryPropertyFlags properties) const {
		vk::ImageCreateInfo imageCreateInfo;
		const glm::uvec2 extent = texData.extent();
		imageCreateInfo.setImageType(vk::ImageType::e2D)
			.setExtent({ extent.x, extent.y, 1 })
			.setMipLevels(1)
			.setArrayLayers(1)
			.setFormat(vk::Format(texData.format()))
			.setTiling(vk::ImageTiling::eOptimal)
			.setInitialLayout(vk::ImageLayout::eUndefined)
			.setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
			.setSharingMode(vk::SharingMode::eExclusive)
			.setSamples(vk::SampleCountFlagBits::e1);

		vk::Image image = device.createImage(imageCreateInfo, nullptr);

		const vk::MemoryRequirements memoryRequirements = device.getImageMemoryRequirements(image);

		vk::MemoryAllocateInfo allocateInfo;
		allocateInfo.setAllocationSize(memoryRequirements.size)
			.setMemoryTypeIndex(findMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));

		vk::DeviceMemory imageMemory = device.allocateMemory(allocateInfo);

		device.bindImageMemory(image, imageMemory, 0);

		return std::make_tuple(image, imageMemory);
	}

	vk::CommandBuffer getSingleCommandBuffer() const {
		const vk::CommandBufferAllocateInfo allocateInfo = vk::CommandBufferAllocateInfo()
			.setLevel(vk::CommandBufferLevel::ePrimary)
			.setCommandPool(commandPool)
			.setCommandBufferCount(1);

		vk::CommandBuffer buffer;
		device.allocateCommandBuffers(&allocateInfo, &buffer);

		vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo()
			.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		buffer.begin(beginInfo);

		return buffer;
	}

	void endSingleCommandBuffer(vk::CommandBuffer buffer) const {
		buffer.end();

		vk::SubmitInfo submitInfo = vk::SubmitInfo()
			.setCommandBufferCount(1)
			.setPCommandBuffers(&buffer);
		graphicsQueue.submit(1, &submitInfo, nullptr);
		graphicsQueue.waitIdle();

		device.freeCommandBuffers(commandPool, 1, &buffer);
	}

	void transitionImageLayout(vk::Image image,
							   vk::Format format,
							   vk::ImageLayout oldLayout,
							   vk::ImageLayout newLayout) const {
		auto buffer = getSingleCommandBuffer();

		vk::ImageMemoryBarrier barrier;
		barrier.setOldLayout(oldLayout)
			.setNewLayout(newLayout)
			.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setImage(image)
			.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 })
			.setSrcAccessMask({})
			.setDstAccessMask({});

		vk::PipelineStageFlags sourceStage, destinationStage;

		// todo: refactor this when more cases are needed
		if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
			barrier.setSrcAccessMask({})
				.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;
		} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal 
			&& newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
			barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
				.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		}

		buffer.pipelineBarrier(sourceStage, destinationStage, {}, 0, nullptr, 0, nullptr, 1, &barrier);

		endSingleCommandBuffer(buffer);
	}

	void copyBufferToImage(const vk::Buffer buffer, const vk::Image image, const glm::uvec2 dim) const {
		const auto commandBuffer = getSingleCommandBuffer();

		vk::BufferImageCopy imageCopy;
		imageCopy.setBufferOffset(0)
			.setBufferRowLength(0)
			.setBufferImageHeight(0)
			.setImageSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 })
			.setImageOffset({ 0, 0, 0 })
			.setImageExtent({ dim.x, dim.y, 1 });

		commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, { imageCopy });

		endSingleCommandBuffer(commandBuffer);
	}

	void createTextureImageView() {
		textureImageView = createImageView(textureImage, textureImageFormat);
	}

	vk::ImageView createImageView(vk::Image image, vk::Format format) const {
		vk::ImageViewCreateInfo viewCreateInfo;
		viewCreateInfo.setImage(image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(format)
			.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

		return device.createImageView(viewCreateInfo);
	}

	void createTextureSampler() {
		vk::SamplerCreateInfo samplerCreateInfo;
		samplerCreateInfo.setMagFilter(vk::Filter::eLinear)
			.setMinFilter(vk::Filter::eLinear)
			.setAddressModeU(vk::SamplerAddressMode::eRepeat)
			.setAddressModeV(vk::SamplerAddressMode::eRepeat)
			.setAddressModeW(vk::SamplerAddressMode::eRepeat)
			.setAnisotropyEnable(true)
			.setMaxAnisotropy(16)
			.setBorderColor(vk::BorderColor::eIntOpaqueBlack)
			.setUnnormalizedCoordinates(false)
			.setCompareEnable(false)
			.setCompareOp(vk::CompareOp::eAlways)
			.setMipmapMode(vk::SamplerMipmapMode::eLinear)
			.setMipLodBias(0.0f)
			.setMinLod(0.0f)
			.setMaxLod(0.0f);

		// Note that this doesn't directly reference the image
		textureSampler = device.createSampler(samplerCreateInfo);
	}

	#pragma endregion Vulkan utility functions

	#pragma region

	void printAllReports() const {
		printExtensionReport();
		printInstanceLayerReport();
		printPhysicalDeviceReport();
		printQueueFamilyReport();
		printMemoryReport();
		printSwapChainReport();
	}

	#pragma region

	void _printLine(const std::vector<int>& alignments) const {
		std::cout << "+";

		for (const auto& field : alignments) {
			for (int i = 0; i < field; i++) {
				std::cout << "-";
			}
			std::cout << "+";
		}
		std::cout << std::endl;
	}

	void printRow(const std::vector<std::string>& fields, const std::vector<int>& alignments) const {
		std::cout << "|";

		for(uint32_t i = 0; i < fields.size(); i++) {
			const std::string& field = fields[i];
			const int& width = alignments[i];
			std::cout << std::left << std::setw(width) << std::setfill(' ') << field << "|";
		}
		std::cout << std::endl;
	}

	void printHeader(const std::vector<std::string>& fields, const std::vector<int>& alignments) const {
		_printLine(alignments);
		printRow(fields, alignments);
		_printLine(alignments);
	}

	void printSection(const std::string& sectionName, const int level = 1) const {
		std::cout << std::endl;
		for (int i = 0; i < level; i++) {
			std::cout << "==";
		}

		std::cout << sectionName << std::endl;
	}

	#pragma endregion Table printing helpers

	void printSwapChainReport() const {
		const SwapChainSupportDetails supportDetails = querySwapChainSupport(physicalDevice);
		printSection("SWAP CHAIN REPORT");
		printSection("EXTENT REPORT", 2);

		const auto extent = pickSwapExtent(supportDetails.capabilities);

		std::cout << "Current Extent: " << extent.width << "*" << extent.height << " w*h px" << std::endl;

		printSection("FORMAT REPORT", 2);
		printSection("FORMATS SUPPORTED", 3);

		const std::vector<int> supportReportAlignments = { 40, 40 };

		printHeader({ "Format", "Color Space" }, supportReportAlignments);
		for (const auto& format : supportDetails.formats) {
			printRow({ to_string(format.format), to_string(format.colorSpace) }, supportReportAlignments);
		}
		_printLine(supportReportAlignments);

		printSection("PRESENT MODES SUPPORTED", 3);
		for (const auto& presentationMode : supportDetails.presentModes) {
			std::cout << "   (*) " << to_string(presentationMode) << std::endl;
		}
	}

	void printMemoryReport() const {
		const std::vector<int> alignments = { 10, 20, 60 };
		const auto memProps = physicalDevice.getMemoryProperties();

		printSection("MEMORY REPORT");
		printSection("MEMORY TYPES REPORT", 2);
		printHeader({ "Index", "Heap", "Flags" }, alignments);
		for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
			const auto& currentType = memProps.memoryTypes[i];
			printRow({
					std::to_string(i), 
					std::to_string(currentType.heapIndex), 
					to_string(currentType.propertyFlags) 
				}, alignments);
		}
		_printLine(alignments);

		printSection("MEMORY HEAPS REPORT", 2);
		printHeader({ "Index", "Size", "Flags" }, alignments);
		for(uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
			const auto& currentHeap = memProps.memoryHeaps[i];
			printRow({
					std::to_string(i),
					std::to_string(currentHeap.size),
					to_string(currentHeap.flags)
				}, alignments);
		}
		_printLine(alignments);
	}

	void printQueueFamilyReport() const {
		std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

		printSection("QUEUE FAMILY REPORT");

		const std::vector<int> alignments = { 20, 60 };

		printHeader({ "QueueCount", "QueueFlags" }, alignments);

		for (const auto& queueFamily : queueFamilies) {
			printRow({ std::to_string(queueFamily.queueCount), to_string(queueFamily.queueFlags) }, alignments);
		}

		_printLine(alignments);
	}

	void printPhysicalDeviceReport() const {
		const std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

		printSection("PHYSICAL DEVICES REPORT");

		std::cout << "Number of devices found: " << devices.size() << std::endl;

		const std::vector<int> alignments = { 30, 30, 20, 20 };

		printHeader({ "Name", "Type", "Vendor", "API Version" }, alignments);

		for (const auto& iPhysicalDevice: devices) {
			const vk::PhysicalDeviceProperties deviceProperties = iPhysicalDevice.getProperties();
			const vk::PhysicalDeviceFeatures deviceFeatures = iPhysicalDevice.getFeatures();

			printRow({
				deviceProperties.deviceName,
				to_string(deviceProperties.deviceType),
				std::to_string(deviceProperties.vendorID),
				std::to_string(deviceProperties.apiVersion)
				}, alignments);
		}

		_printLine(alignments);
	}

	void printInstanceLayerReport() const {
		std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

		printSection("INSTANCE LAYER REPORT");

		const std::vector<int> alignments = { 50, 60 };
		
		printHeader({ "Layer Name", "Description" }, alignments);

		for (const auto layerProperties : availableLayers) {
			printRow({ layerProperties.layerName, layerProperties.description }, alignments);
		}

		_printLine(alignments);
	}

	void printExtensionReport() const {
		std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();
		std::vector<std::string> requiredExtensions = getRequiredExtensions();
		std::set<std::string> extensionSet(requiredExtensions.begin(), requiredExtensions.end());

		printSection("EXTENSION REPORT");

		std::cout << "Extensions supported: " << extensions.size() << std::endl;
		std::cout << "Extensions required: " << requiredExtensions.size() << std::endl;

		const std::vector<int> alignments = { 60, 20 };

		printHeader({ "Name", "Version" }, alignments);

		for (const auto extension : extensions) {
			extensionSet.erase(extension.extensionName);
			printRow({ extension.extensionName, std::to_string(extension.specVersion) }, alignments);
		}

		_printLine(alignments);

		std::cout << "Required extensions supported: " << (extensionSet.empty() ? "TRUE" : "FALSE") << std::endl;
	}

	#pragma endregion Debug Info Reports
};

int main() {
	ApplicationModule app;

	try {
		app.run();
	} catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		std::cout << "Press enter to acknowledge and quit...";
		std::cin.get();
		return EXIT_FAILURE;
	}

#ifdef MSVC
	std::cout << "Press enter to quit...";
	std::cin.get();
#endif

	return EXIT_SUCCESS;
}


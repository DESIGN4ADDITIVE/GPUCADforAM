

/*
   Reference  - https://vulkan-tutorial.com/
*/

#include <unistd.h>
#include <typeinfo>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <functional>
#include <set>
#include <string.h>
#include <limits>

#include "VulkanBaseApp.h"

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>


#ifndef countof
#define countof(x) (sizeof(x) / sizeof(*(x)))
#endif


uint32_t graphic_queue_index;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}
static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}
    
bool show_demo_window = true;
bool show_another_window = true;
bool update_shader_buffer = true;

int VulkanBaseApp::screen_width = 500;
int VulkanBaseApp::screen_height = 500;





static const char *validationLayers[] = { "VK_LAYER_KHRONOS_validation" };
static const size_t MAX_FRAMES_IN_FLIGHT = 2;



struct {
 
    VkPipeline graphicsPipeline;
    VkPipeline graphicsPipelineread;
    VkPipeline graphicsPipelineone;
    VkPipeline graphicsPipelineoneread;
    
} pipelines;


void VulkanBaseApp::resizeCallback(GLFWwindow *window, int width, int height)
{
    VulkanBaseApp *app = reinterpret_cast<VulkanBaseApp *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}



VulkanBaseApp::VulkanBaseApp(const std::string& appName, bool enableValidation) :
    appName(appName),
    enableValidation(enableValidation),
    vpcount(1),
    instance(VK_NULL_HANDLE),
    window(nullptr),
    debugMessenger(VK_NULL_HANDLE),
    surface(VK_NULL_HANDLE),
    physicalDevice(VK_NULL_HANDLE),
    device(VK_NULL_HANDLE),
    graphicsQueue(VK_NULL_HANDLE),
    presentQueue(VK_NULL_HANDLE),
    swapChain(VK_NULL_HANDLE),
    vkDeviceUUID(),
    swapChainImages(),
    swapChainFormat(),
    swapChainExtent(),
    swapChainImageViews(),
    shaderFiles(),
    shaderFilesread(),
    shaderFilesone(),
    shaderFilesoneread(),
    renderPass(),
    pipelineLayout(VK_NULL_HANDLE),
    pipelineLayoutread(VK_NULL_HANDLE),
    swapChainFramebuffers(),
    viewports(),
    scissors(),
    commandPool(VK_NULL_HANDLE),
    commandBuffers(),
    imageAvailableSemaphores(),
    renderFinishedSemaphores(),
    inFlightFences(),
    storageBuffers(),
    storageMemory(),
    uniformBuffers(),
    uniformMemory(),
    descriptorSetLayout(VK_NULL_HANDLE),
    descriptorSetLayoutread(VK_NULL_HANDLE),
    descriptorPool(VK_NULL_HANDLE),
    descriptorSets(),
    descriptorSetsread(),
    colorImages(),
    colorImageMemory(),
    colorImageViews(),
    depthImages(),
    depthImageMemory(),
    depthImageViews(),
    currentFrame(0),
    framebufferResized(false),
    show_mesh(true),
    show_grid(true),
    shift(false),
    reset(false),
    retain(false),
    undoo(false),
    calculate(true),
    topo_data(false)
    {
     
    }

VkExternalSemaphoreHandleTypeFlagBits VulkanBaseApp::getDefaultSemaphoreHandleType()
{

    return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

}

VkExternalMemoryHandleTypeFlagBits VulkanBaseApp::getDefaultMemHandleType()
{

    return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

}

VulkanBaseApp::~VulkanBaseApp()
{
  
    ImGui_ImplVulkan_Shutdown();

    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();
    
    cleanupSwapChain();

    clean_up();


    cleanupSyncObjects();

    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }

    if (enableValidation) {
        PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance, debugMessenger, nullptr);
        }
    }

    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }

    if (window) {
        glfwDestroyWindow(window);
    }
  

    glfwTerminate();
}

void VulkanBaseApp::init()
{
    initWindow();
    initVulkan();
    initImgui();
   
}

VkCommandBuffer VulkanBaseApp::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanBaseApp::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanBaseApp::initWindow()
{
    
    glfwInit();


    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    videoMode = glfwGetVideoMode(primaryMonitor);

    VulkanBaseApp::screen_width = videoMode->width * 0.8;
    VulkanBaseApp::screen_height = videoMode->height *0.8;

    ImguiApp::window_extent = {videoMode->width * 0.3f,videoMode->height * 0.6f};

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(screen_width,screen_height , appName.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, resizeCallback);
    glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GL_FALSE);
   

    
}


std::vector<const char *> VulkanBaseApp::getRequiredExtensions() const
{
    return std::vector<const char *>();
}

std::vector<const char *> VulkanBaseApp::getRequiredDeviceExtensions() const
{
    return std::vector<const char *>();
}

void VulkanBaseApp::initVulkan()
{
    createInstance();
    createSurface();
    createDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createDescriptorSetLayoutread();
    createGraphicsPipeline();
    createCommandPool();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    initVulkanCuda_semaphores();
    createStorageBuffers((150*150*150));
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createDescriptorSetsread();
    createCommandBuffers();
    createSyncObjects();

}


void VulkanBaseApp::initImgui()
{       //// Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); 
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
        io.Fonts->AddFontFromFileTTF("../src/fonts/Cousine-Regular.ttf",25);
        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.24f, 0.24f, 1.00f);
        
        ImGui_ImplGlfw_InitForVulkan(window,true);

        ImGui_ImplVulkan_InitInfo init_info;

        init_info.Instance = instance;
        init_info.PhysicalDevice = physicalDevice;
        init_info.Device = device;
        init_info.QueueFamily = graphic_queue_index;
        init_info.Queue = graphicsQueue;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = descriptorPool;
        init_info.RenderPass = renderPass;
        init_info.Subpass = 1;
        init_info.MinImageCount = 2;
        init_info.ImageCount = MAX_FRAMES_IN_FLIGHT;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = check_vk_result;
        init_info.MinAllocationSize = 1024*1024;
        ImGui_ImplVulkan_Init(&init_info);

}






static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("Failed to find supported format!");
}

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return ~0;
}

static bool supportsValidationLayers()
{
    std::vector<VkLayerProperties> availableLayers;
    uint32_t layerCount;

    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    availableLayers.resize(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char * layerName : validationLayers) {
        bool layerFound = false;

        for (const auto & layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}



void VulkanBaseApp::createInstance()
{
    if (enableValidation && !supportsValidationLayers()) {
        throw std::runtime_error("Validation requested, but not supported!");
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    std::vector<const char *> exts = getRequiredExtensions();

    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        
        exts.insert(exts.begin(), glfwExtensions, glfwExtensions + glfwExtensionCount);

       
        if (enableValidation) {
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
           
        }
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    createInfo.ppEnabledExtensionNames = exts.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
    if (enableValidation) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(countof(validationLayers));
        createInfo.ppEnabledLayerNames = validationLayers;

        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;
        

        createInfo.pNext = &debugCreateInfo;

        //////////////////////////////////////////////////////////////////////////////////////////////

        VkValidationFeaturesEXT validationFeatures = {};
        validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validationFeatures.enabledValidationFeatureCount = 1;

        VkValidationFeatureEnableEXT enabledValidationFeatures[1] = {
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};

        validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;
    
        validationFeatures.pNext = createInfo.pNext;
        createInfo.pNext = &validationFeatures;
     

    }
    else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }
    
    if (enableValidation) {
        PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func == nullptr || func(instance, &debugCreateInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug messenger!");
        }

    }
}

void VulkanBaseApp::createSurface()
{
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

static bool findGraphicsQueueIndicies(VkPhysicalDevice device, VkSurfaceKHR surface, uint32_t& graphicsFamily, uint32_t& presentFamily)
{
    uint32_t queueFamilyCount = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    graphicsFamily = presentFamily = ~0;

    for (uint32_t i = 0; i < queueFamilyCount; i++) {

        if (queueFamilies[i].queueCount > 0) {
            if (graphicsFamily == ~0 && queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsFamily = i;
            }
            uint32_t presentSupport = 0;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentFamily == ~0 && presentSupport) {
                presentFamily = i;
            }
            if (presentFamily != ~0 && graphicsFamily != ~0) {
                break;
            }
        }
    }

    return graphicsFamily != ~0 && presentFamily != ~0;
}

static bool hasAllExtensions(VkPhysicalDevice device, const std::vector<const char *>& deviceExtensions)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto & extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

static void getSwapChainProperties(VkPhysicalDevice device, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR& capabilities, std::vector<VkSurfaceFormatKHR>& formats, std::vector<VkPresentModeKHR>& presentModes)
{
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, formats.data());
    }
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, presentModes.data());
    }
}


void VulkanBaseApp::createDevice()
{
    {
        
        uint32_t deviceCount = 0;
      
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find Vulkan capable GPUs!");
        }
        
        
        std::vector<VkPhysicalDevice> phyDevs(deviceCount);

        VkResult str  = vkEnumeratePhysicalDevices(instance, &deviceCount, phyDevs.data());
       
        for (const auto& device : phyDevs)
        {
        auto props = VkPhysicalDeviceProperties{};
        vkGetPhysicalDeviceProperties(device, &props);
        
        if (props.deviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            
            physicalDevice = device;
            break;
        }

        else if (props.deviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        {
            printf("Not selecting Integrated GPU \n\n");
        }
        else
        {
            printf("No suitable device found! \n\n");
        }
       
        }

        if (physicalDevice == VK_NULL_HANDLE){

            throw std::runtime_error(" Failed to set Physical device");
        }

    }

    uint32_t graphicsQueueIndex, presentQueueIndex;
    findGraphicsQueueIndicies(physicalDevice, surface, graphicsQueueIndex, presentQueueIndex);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueFamilyIndices = { graphicsQueueIndex, presentQueueIndex };

    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueFamilyIndices) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = graphicsQueueIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceFeatures.fillModeNonSolid = true;
    //myline
    deviceFeatures.geometryShader = true;
    deviceFeatures.multiViewport = true;
    deviceFeatures.wideLines = true;
    deviceFeatures.shaderTessellationAndGeometryPointSize = true;
    deviceFeatures.fragmentStoresAndAtomics = true;
    deviceFeatures.vertexPipelineStoresAndAtomics = true;
    
    

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    std::vector<const char *> deviceExtensions = getRequiredDeviceExtensions();
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidation) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(countof(validationLayers));
        createInfo.ppEnabledLayerNames = validationLayers;
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, graphicsQueueIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentQueueIndex, 0, &presentQueue);

    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;
    

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
    fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
    if (fpGetPhysicalDeviceProperties2 == NULL) {
        throw std::runtime_error("Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not found.\n");
    }

    fpGetPhysicalDeviceProperties2(physicalDevice, &vkPhysicalDeviceProperties2);

    memcpy(vkDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID,  VK_UUID_SIZE);

    
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
        return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    }

    for (const auto & availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto & availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
        else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

static VkExtent2D chooseSwapExtent(GLFWwindow *window, const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

void VulkanBaseApp::createSwapChain()
{
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR format;
    VkPresentModeKHR presentMode;
    VkExtent2D extent;
    uint32_t imageCount;

    {
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
        getSwapChainProperties(physicalDevice, surface, capabilities, formats, presentModes);
        format = chooseSwapSurfaceFormat(formats);
        presentMode = chooseSwapPresentMode(presentModes);
        extent = chooseSwapExtent(window, capabilities);
        imageCount = MAX_FRAMES_IN_FLIGHT;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }
    }
    
    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = format.format;
    createInfo.imageColorSpace = format.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ;

    uint32_t queueFamilyIndices[2];
    findGraphicsQueueIndicies(physicalDevice, surface, queueFamilyIndices[0], queueFamilyIndices[1]);

    if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = countof(queueFamilyIndices);
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainFormat = format.format;
    swapChainExtent = extent;
}

static VkImageView createImageView(VkDevice dev, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
    VkImageView imageView;
    VkImageViewCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = aspectFlags;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(dev, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image views!");
    }

    return imageView;
}

static void createImage(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
    VkImageLayout imageLayout;
   
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
   
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

void VulkanBaseApp::createImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());

    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
        swapChainImageViews[i] = createImageView(device, swapChainImages[i], swapChainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}



void VulkanBaseApp::createRenderPass()
{
    
    VkAttachmentDescription swap_colorAttachment = {};
    swap_colorAttachment.format = swapChainFormat;
    swap_colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    swap_colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    swap_colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    swap_colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    swap_colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    swap_colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swap_colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference swap_colorAttachmentRef = {};
    swap_colorAttachmentRef.attachment = 0;
    swap_colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    

    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = VK_FORMAT_B8G8R8A8_UNORM;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 1;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = findSupportedFormat(physicalDevice,
    { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
    VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 2;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription attachments[] = {swap_colorAttachment,colorAttachment ,depthAttachment};

    std::array<VkSubpassDescription,2> subpassDescriptions{};

    subpassDescriptions[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescriptions[0].colorAttachmentCount = 1;
    subpassDescriptions[0].pColorAttachments = &colorAttachmentRef;
    subpassDescriptions[0].pDepthStencilAttachment = &depthAttachmentRef;


    VkAttachmentReference inputReferences[2]{};
    inputReferences[0].attachment = 1;
    inputReferences[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    inputReferences[1].attachment = 2;
    inputReferences[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    subpassDescriptions[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescriptions[1].colorAttachmentCount = 1;
    subpassDescriptions[1].pColorAttachments = &swap_colorAttachmentRef;
    
    subpassDescriptions[1].inputAttachmentCount = 2;
    subpassDescriptions[1].pInputAttachments = inputReferences;

    VkSubpassDependency dependency_0 = {};
    dependency_0.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency_0.dstSubpass = 0;

    dependency_0.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependency_0.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT ;

    dependency_0.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT ;
    dependency_0.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT ;

    dependency_0.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


     VkSubpassDependency dependency_1 = {};
    dependency_1.srcSubpass = 0;
    dependency_1.dstSubpass = 1;
    dependency_1.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency_1.dstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    dependency_1.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency_1.dstAccessMask = 0;

    VkSubpassDependency dependency_2 = {};
    dependency_2.srcSubpass = 0;
    dependency_2.dstSubpass = 1;
    dependency_2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency_2.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

    dependency_2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency_2.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
    dependency_2.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    
    dependency_2.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkSubpassDependency dependency_3 = {};
    dependency_3.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency_3.dstSubpass = 1;
    
    dependency_3.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependency_3.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency_3.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependency_3.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; 
    dependency_3.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


    VkSubpassDependency dependencies[]= {dependency_0, dependency_1, dependency_2, dependency_3};
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = countof(attachments);
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = subpassDescriptions.size();
    renderPassInfo.pSubpasses = subpassDescriptions.data();
    renderPassInfo.dependencyCount = countof(dependencies);
    renderPassInfo.pDependencies = dependencies;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void VulkanBaseApp::createDescriptorSetLayout()
{
    
    VkDescriptorSetLayoutBinding   uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;


    VkDescriptorSetLayoutBinding   uboLayoutBindingStorage = {};
    uboLayoutBindingStorage.binding = 1;
    uboLayoutBindingStorage.descriptorCount = 1;
    uboLayoutBindingStorage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    uboLayoutBindingStorage.pImmutableSamplers = nullptr;
    uboLayoutBindingStorage.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;


    std::vector<VkDescriptorSetLayoutBinding> ubo_binding = { uboLayoutBinding, uboLayoutBindingStorage};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ubo_binding.size();
    layoutInfo.pBindings = ubo_binding.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }


}


void VulkanBaseApp::createDescriptorSetLayoutread()
{

    VkDescriptorSetLayoutBinding   uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;

    VkDescriptorSetLayoutBinding   uboLayoutstorageBinding = {};
    uboLayoutstorageBinding.binding = 1;
    uboLayoutstorageBinding.descriptorCount = 1;
    uboLayoutstorageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    uboLayoutstorageBinding.pImmutableSamplers = nullptr;
    uboLayoutstorageBinding.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    
    VkDescriptorSetLayoutBinding   uboLayoutcolorBinding = {};
    uboLayoutcolorBinding.binding = 2;
    uboLayoutcolorBinding.descriptorCount = 1;
    uboLayoutcolorBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    uboLayoutcolorBinding.pImmutableSamplers = nullptr;
    uboLayoutcolorBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
   
    VkDescriptorSetLayoutBinding   uboLayoutdepthBinding = {};
    uboLayoutdepthBinding.binding = 3;
    uboLayoutdepthBinding.descriptorCount = 1;
    uboLayoutdepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    uboLayoutdepthBinding.pImmutableSamplers = nullptr;
    uboLayoutdepthBinding.stageFlags =  VK_SHADER_STAGE_FRAGMENT_BIT ;

    std::vector<VkDescriptorSetLayoutBinding> ubo_bindingg = {uboLayoutBinding, uboLayoutstorageBinding, uboLayoutcolorBinding, uboLayoutdepthBinding  };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ubo_bindingg.size();
    layoutInfo.pBindings = ubo_bindingg.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayoutread) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layoutread!");
    }
}




VkShaderModule createShaderModule(VkDevice device, const char *filename)
{
    std::vector<char> shaderContents;
    std::ifstream shaderFile(filename, std::ios_base::in | std::ios_base::binary);
    VkShaderModuleCreateInfo createInfo = {};
    VkShaderModule shaderModule;
    
    if (!shaderFile.good()) {
        throw std::runtime_error("Failed to load shader contents");
    }
    readFile(shaderFile, shaderContents);

    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = shaderContents.size();
 
    
    createInfo.pCode = reinterpret_cast<const uint32_t *>(shaderContents.data());

    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shaderModule;
}

void VulkanBaseApp::createStorageBuffers(size_t nVerts)
{

}
void VulkanBaseApp::erase_topo_data()
{

}

void VulkanBaseApp::erase_lattice_data()
{

}

void VulkanBaseApp::erase_primitive_data()
{

}

void VulkanBaseApp::getVertexDescriptions(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc)
{
}

void VulkanBaseApp::getVertexDescriptionsone(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc)
{
}



void VulkanBaseApp::getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info)
{

}

void VulkanBaseApp::createGraphicsPipeline()
{
    
    
    std::vector<VkPipelineShaderStageCreateInfo> shaderStageInfos(shaderFiles.size());
    for (size_t i = 0; i < shaderFiles.size(); i++) {
        shaderStageInfos[i] = {};
        shaderStageInfos[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfos[i].stage = shaderFiles[i].first;
        shaderStageInfos[i].module = createShaderModule(device, shaderFiles[i].second.c_str());
        shaderStageInfos[i].pName = "main";
    }

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};

    std::vector<VkVertexInputBindingDescription> vertexBindingDescriptions;
    std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions;

    getVertexDescriptions(vertexBindingDescriptions, vertexAttributeDescriptions);

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptions.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    getAssemblyStateInfo(inputAssembly);
    
    viewports.resize(vpcount);
    scissors.resize(vpcount);
  
    viewports[0] = {screen_width * 0.4f, 0, screen_height * 1.0f, screen_height * 1.0f, 0.0f, 1.0f };
    scissors[0] = {{ int(screen_width * 0.4f), 0 },{uint(screen_height * 1.0f),uint(screen_height * 1.0f)}};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewports[0];
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissors[0];

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.lineWidth = 2.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  
    multisampling.flags = 0;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;


    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = 0xf;
    colorBlendAttachment.blendEnable = VK_FALSE;


    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
 
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;


    ////////////////////////Push Constants//////////////////////////////////
    VkPushConstantRange push_constant;
    push_constant.offset = 0;
    push_constant.size = sizeof(push_constants);
    push_constant.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT ;


    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1; // Optional
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
    pipelineLayoutInfo.pPushConstantRanges = &push_constant; // Optional

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStageInfos.size());
    pipelineInfo.pStages = shaderStageInfos.data();

    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;


    pipelineInfo.layout = pipelineLayout;

    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;


    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines.graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    for (size_t i = 0; i < shaderStageInfos.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfos[i].module, nullptr);
    }
    

    //////////////////////////////////////////shaderFilesone////////////////////////////////////////////////
    
    for (size_t i = 0; i < shaderFilesone.size(); i++) {

        shaderStageInfos[i].module = createShaderModule(device, shaderFilesone[i].second.c_str());

    }


    std::vector<VkVertexInputBindingDescription> vertexBindingDescriptionsone;
    std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptionsone;
    getVertexDescriptionsone(vertexBindingDescriptionsone, vertexAttributeDescriptionsone);

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptionsone.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptionsone.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptionsone.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptionsone.data();

    rasterizer.lineWidth = 1.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;


    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines.graphicsPipelineone) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline one!");
    }

    for (size_t i = 0; i < shaderStageInfos.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfos[i].module, nullptr);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////


    std::vector<VkPipelineShaderStageCreateInfo> shaderStageInfosread(shaderFilesread.size());
    for (size_t i = 0; i < shaderFilesread.size(); i++) {


        shaderStageInfosread[i] = {};
        shaderStageInfosread[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfosread[i].stage = shaderFilesread[i].first;
        shaderStageInfosread[i].module = createShaderModule(device, shaderFilesread[i].second.c_str());
        shaderStageInfosread[i].pName = "main";

    }
    

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptions.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions.data();


    rasterizer.lineWidth = 2.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkPipelineLayoutCreateInfo pipelineLayoutInforead = {};
    pipelineLayoutInforead.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInforead.setLayoutCount = 1; // Optional
    pipelineLayoutInforead.pSetLayouts = &descriptorSetLayoutread; 

    VkPushConstantRange push_constantone;
    push_constantone.offset = 0;
    push_constantone.size = sizeof(push_constants);
    push_constantone.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ;

    pipelineLayoutInforead.pushConstantRangeCount = 1; // Optional
    pipelineLayoutInforead.pPushConstantRanges = &push_constantone; // Optional

    if (vkCreatePipelineLayout(device, &pipelineLayoutInforead, nullptr, &pipelineLayoutread) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout read");
    }

    VkGraphicsPipelineCreateInfo pipelineInforead = {};
    pipelineInforead.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInforead.stageCount = static_cast<uint32_t>(shaderStageInfosread.size());
    pipelineInforead.pStages = shaderStageInfosread.data();

    pipelineInforead.pVertexInputState = &vertexInputInfo;
    pipelineInforead.pInputAssemblyState = &inputAssembly;
 
    pipelineInforead.pViewportState = &viewportState;
    pipelineInforead.pRasterizationState = &rasterizer;
    pipelineInforead.pMultisampleState = &multisampling;

    depthStencil.depthWriteEnable = VK_FALSE;
    pipelineInforead.pDepthStencilState = &depthStencil; // Optional
    colorBlending.attachmentCount = 1;
    pipelineInforead.pColorBlendState = &colorBlending;

    pipelineInforead.layout = pipelineLayoutread;

    pipelineInforead.renderPass = renderPass;
    pipelineInforead.subpass = 1;


    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInforead, nullptr, &pipelines.graphicsPipelineread) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    for (size_t i = 0; i < shaderStageInfosread.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfosread[i].module, nullptr);
    }

    /////////////////////////////////////////shaderFilesoneread////////////////////////////////////////////

    for (size_t i = 0; i < shaderFilesoneread.size(); i++) {

        shaderStageInfosread[i].module = createShaderModule(device, shaderFilesoneread[i].second.c_str());

    }

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptionsone.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptionsone.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptionsone.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptionsone.data();

    rasterizer.lineWidth = 1.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;


    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    viewportState.pViewports = &viewports[0];
    viewportState.pScissors = &scissors[0];
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInforead, nullptr, &pipelines.graphicsPipelineoneread) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline one!");
    }

    for (size_t i = 0; i < shaderStageInfosread.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfosread[i].module, nullptr);
    }


}



void VulkanBaseApp::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());
    VkImageView attachments[3];
    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = countof(attachments);
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = swapChainExtent.width;
    framebufferInfo.height = swapChainExtent.height;
    framebufferInfo.layers = 1;
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        
        attachments[0] = swapChainImageViews[i] ;
        attachments[1] = colorImageViews[i];
        attachments[2] = depthImageViews[i];
        
        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void VulkanBaseApp::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo = {};
    uint32_t graphicsIndex, presentIndex;

    findGraphicsQueueIndicies(physicalDevice, surface, graphicsIndex, presentIndex);

    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Optional

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }
}

void VulkanBaseApp::createDepthResources()
{
    
    depthImages.resize(swapChainImages.size());
    depthImageMemory.resize(swapChainImages.size());
    depthImageViews.resize(swapChainImages.size());
    VkFormat depthFormat = findSupportedFormat(physicalDevice,
    { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
    VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    for(int i = 0; i < swapChainImages.size(); i++)
    {
        createImage(physicalDevice, device, swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT , VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImages[i], depthImageMemory[i]);
        depthImageViews[i] = createImageView(device, depthImages[i], depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT );
    }
}  

void VulkanBaseApp::createColorResources()
{
    colorImages.resize(swapChainImages.size());
    colorImageMemory.resize(swapChainImages.size());
    colorImageViews.resize(swapChainImages.size());
    VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
    for(int i = 0; i < swapChainImages.size(); i++)
    {
        createImage(physicalDevice, device, swapChainExtent.width, swapChainExtent.height, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT , VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImages[i], colorImageMemory[i]);
        colorImageViews[i] = createImageView(device, colorImages[i], colorFormat, VK_IMAGE_ASPECT_COLOR_BIT );
    }

}




void VulkanBaseApp::createUniformBuffers()
{
    VkDeviceSize size = getUniformSize();
    if (size > 0) {
        uniformBuffers.resize(swapChainImages.size());
        uniformMemory.resize(swapChainImages.size());
        for (size_t i = 0; i < uniformBuffers.size(); i++) {
            createBuffer(getUniformSize(),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformMemory[i]);
        }
    }
}

void VulkanBaseApp::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 5> poolSizes{};

    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    poolSizes[2].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    poolSizes[3].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    poolSizes[3].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);


    poolSizes[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[4].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT *4);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)*4;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}


void VulkanBaseApp::createDescriptorSets()
{
    
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;


    descriptorSets.resize(swapChainImages.size());

    for(int i = 0; i < swapChainImages.size();i++)
    {
        
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets[i]) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.offset = 0;
        bufferInfo.range = getUniformSize();
        bufferInfo.buffer = uniformBuffers[i];

        VkWriteDescriptorSet descriptorWrite = {};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;


        VkDescriptorBufferInfo bufferInfo_1 = {};
        bufferInfo_1.offset = 0;
        bufferInfo_1.range = (150 * 150 * 150 * 4);
        bufferInfo_1.buffer = storageBuffers[i];

        VkWriteDescriptorSet descriptorWriteone = {};
        descriptorWriteone.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteone.dstSet = descriptorSets[i];
        descriptorWriteone.dstBinding = 1;
        descriptorWriteone.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWriteone.descriptorCount = 1;
        descriptorWriteone.pBufferInfo = &bufferInfo_1;

        std::vector<VkWriteDescriptorSet> descriptorWrites_sets = {descriptorWrite, descriptorWriteone};
        vkUpdateDescriptorSets(device, descriptorWrites_sets.size(), descriptorWrites_sets.data(), 0, nullptr);

    }

    
}

void VulkanBaseApp::createDescriptorSetsread()
{
    
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayoutread;

    descriptorSetsread.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) 
    {
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSetsread[i]) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.offset = 0;
        bufferInfo.range = getUniformSize();
        bufferInfo.buffer = uniformBuffers[i];

        VkWriteDescriptorSet descriptorWrite = {};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstSet = descriptorSetsread[i];
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        VkDescriptorBufferInfo bufferInfoone = {};
        bufferInfoone.offset = 0;
        bufferInfoone.range = (150 * 150 * 150 * 4);
        bufferInfoone.buffer = storageBuffers[i];

        VkWriteDescriptorSet descriptorWriteone = {};
        descriptorWriteone.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteone.dstSet = descriptorSetsread[i];
        descriptorWriteone.dstBinding = 1;
        descriptorWriteone.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWriteone.descriptorCount = 1;
        descriptorWriteone.pBufferInfo = &bufferInfoone;

        VkDescriptorImageInfo ImageInfocolor = {};
        ImageInfocolor.sampler = VK_NULL_HANDLE;
        ImageInfocolor.imageView = colorImageViews[i];
        ImageInfocolor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;


        VkWriteDescriptorSet descriptorWritecolor = {};
        descriptorWritecolor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWritecolor.dstSet = descriptorSetsread[i];
        descriptorWritecolor.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        descriptorWritecolor.dstBinding = 2;
        descriptorWritecolor.pImageInfo = &ImageInfocolor;
        descriptorWritecolor.descriptorCount = 1;

        
        VkDescriptorImageInfo ImageInfodepth = {};
        ImageInfodepth.sampler = VK_NULL_HANDLE;
        ImageInfodepth.imageView = depthImageViews[i];
        ImageInfodepth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;


        VkWriteDescriptorSet descriptorWritedepth = {};
        descriptorWritedepth.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWritedepth.dstSet = descriptorSetsread[i];
        descriptorWritedepth.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        descriptorWritedepth.dstBinding = 3;
        descriptorWritedepth.pImageInfo = &ImageInfodepth;
        descriptorWritedepth.descriptorCount = 1;

        std::vector<VkWriteDescriptorSet> descriptorWrites = {descriptorWrite,descriptorWriteone, descriptorWritecolor, descriptorWritedepth};
        vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

    }

    
}


void VulkanBaseApp::createCommandBuffers()
{
    
    commandBuffers.resize(swapChainFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    VkResult res = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());

    if (res != VK_SUCCESS) {
        
        throw std::runtime_error("failed to allocate command buffers!");
    }
  
}


void VulkanBaseApp::updatecommandBuffers(VkCommandBuffer commandBuffer, uint32_t image_index, uint32_t currentFrame)
{
   
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        
        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[image_index];

        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColors[3];
        clearColors[0].color = {clear_color.x, clear_color.y, clear_color.z, clear_color.w};
        clearColors[1].color = {  0.7f, 0.8f, 0.8f, 1.0f  };
        clearColors[2].depthStencil = {1.0f, 0 };
        renderPassInfo.clearValueCount = countof(clearColors);
        renderPassInfo.pClearValues = clearColors;
       
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
       
		vkCmdSetViewport(commandBuffer, 0,1,&viewports[0]);
      
        vkCmdSetScissor(commandBuffer,0,1,&scissors[0]);
    
        vkCmdSetLineWidth(commandBuffer,2.0);
        
        if(ImguiApp::vulkan_buffer_created)
        {
            if(VulkanBaseApp::show_grid)
            {

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
                if(ImguiApp::lattice )
                {
                    if(ImguiApp::show_unit_lattice_data)
                    {
                        fillRenderingCommandBuffer_unit_lattice(commandBuffer);
                    }
                    else if( ImguiApp::show_lattice_data)
                    {
                        fillRenderingCommandBuffer_spatial_lattice(commandBuffer);
                    }
                }
                else
                {
                    fillRenderingCommandBuffer(commandBuffer);
                }
               
            }

             if(VulkanBaseApp::show_mesh)
            {
        
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipelineone);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
           
                if(ImguiApp::lattice || ImguiApp::show_topo_lattice)
                {
                    if(ImguiApp::show_unit_lattice_data)
                    {
                        fillRenderingCommandBufferfour(commandBuffer);
                    }
                    else
                    {
                        
                        fillRenderingCommandBuffertwo(commandBuffer);
                    }
                }

                else if(ImguiApp::structural || ImguiApp::thermal)
                {
                    fillRenderingCommandBufferone(commandBuffer);
                }
                else if((ImguiApp::primitives))
                {
                    fillRenderingCommandBufferthree(commandBuffer);
                }

            }
          
        }
    
    
        vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
 
        if(ImguiApp::vulkan_buffer_created)
        {
            if(VulkanBaseApp::show_grid)
            {
          
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipelineread);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayoutread, 0, 1, &descriptorSetsread[currentFrame], 0, nullptr);

                if(ImguiApp::lattice )
                {
                    if(ImguiApp::show_unit_lattice_data)
                    {
                        fillRenderingCommandBuffer_unit_lattice_subpass1(commandBuffer);
                    }
                    else if( ImguiApp::show_lattice_data)
                    {
                        fillRenderingCommandBuffer_spatial_lattice_subpass1(commandBuffer);
                    }
                }
                else
                {
                    fillRenderingCommandBuffer_subpass1(commandBuffer);
                }
           

            }

           
            
            if(VulkanBaseApp::show_mesh)
            {
        
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipelineoneread);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayoutread, 0, 1, &descriptorSetsread[currentFrame], 0, nullptr);
        
                if(ImguiApp::lattice || ImguiApp::show_topo_lattice)
                {
                    
                    if(ImguiApp::show_unit_lattice_data)
                    {
                        fillRenderingCommandBufferfour_subpass1(commandBuffer);
                    }
                    else
                    {
                    
                        fillRenderingCommandBuffertwo_subpass1(commandBuffer);
                    }
                }

                else if(ImguiApp::structural || ImguiApp::thermal)
                {
                    fillRenderingCommandBufferone_subpass1(commandBuffer);
                }

                else if((ImguiApp::primitives))
                {
                    
                    fillRenderingCommandBufferthree_subpass1(commandBuffer);
                }
                
            }
        }

      
        ImGui_ImplVulkan_NewFrame();

        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();
        
        if (ImGui::BeginMainMenuBar())
        {
            
            
            if (ImGui::BeginMenu("File  "))
            {
                ImguiApp::make_all_inactive(window_bools);

                ImGui::MenuItem("Export Data",NULL,&export_settings);
                ImguiApp::make_inactive(window_bools,&export_settings);
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Edit  "))
            {
                ImguiApp::make_all_inactive(window_bools);

                if(ImGui::MenuItem("Grid Settings",NULL,&grid_settings))
                {
                    ImguiApp::make_inactive(window_bools,&grid_settings);
                }

                if(ImGui::MenuItem("Background Color ",NULL,&background_color))
                {
                        ImguiApp::make_inactive(window_bools,&background_color);
                }


                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("View  "))
            {
                ImguiApp::make_all_inactive(window_bools);
                
                if(ImGui::MenuItem("View Settings",NULL,&view_settings))
                {
                    ImguiApp::make_inactive(window_bools,&view_settings);
                }

                
                if (ImGui::BeginMenu("VIEWS"))
                {
                    if(ImGui::MenuItem("FRONT",NULL,&view_front))
                    {
                        ImguiApp::make_inactive(view_bools,&view_front);

                    }
                    if(ImGui::MenuItem("BACK",NULL,&view_back))
                    {
                        ImguiApp::make_inactive(view_bools,&view_back);
                    }
                    if(ImGui::MenuItem("TOP",NULL,&view_top))
                    {
                        ImguiApp::make_inactive(view_bools,&view_top);
                    }
                    if(ImGui::MenuItem("BOTTOM",NULL,&view_bottom))
                    {
                        ImguiApp::make_inactive(view_bools,&view_bottom);
                    }
                    if(ImGui::MenuItem("RIGHT",NULL,&view_right))
                    {
                        ImguiApp::make_inactive(view_bools,&view_right);
                    }
                    if(ImGui::MenuItem("LEFT",NULL,&view_left))
                    {
                        ImguiApp::make_inactive(view_bools,&view_left);
                    }
                    ImGui::EndMenu();
                }
                    
                ImGui::EndMenu();
            }


            if (ImGui::BeginMenu("Module  "))
            {
                ImguiApp::make_all_inactive(window_bools);
                
                if(ImGui::MenuItem("PRIMITIVE",NULL,&primitives))
                {
                    
                    
                    if((show_topo_lattice || topo_done_lattice_do ))
                    {

                        show_topo_lattice = false;
                        topo_done_lattice_do = false;
                        erase_topo_data();
                     
                    }
                    if(show_unit_lattice_data || show_lattice_data)
                    {
                        show_lattice_data = false;
                        show_unit_lattice_data = false;
                        erase_lattice_data();
                      
                    }

                    ImguiApp::make_inactive(physics_bools,&primitives);
                }

                else if(ImGui::MenuItem("LATTICE",NULL,&lattice))
                {
                    if(show_model || show_primitive_lattice)
                    {
                        show_model = false;
                        show_primitive_lattice =false;
                        erase_primitive_data();
                    
                    }
                    
                    
                    if((show_topo_lattice || topo_done_lattice_do ))
                    {
                    
                        show_topo_lattice = false;
                        topo_done_lattice_do = false;
                        erase_topo_data();
                       
                    }

                    ImguiApp::make_inactive(physics_bools,&lattice);
                    
                }

                else if(ImGui::BeginMenu("PHYSICS"))
                {
                    if(show_model || show_primitive_lattice)
                    {
                        show_model = false;
                        show_primitive_lattice =false;
                        erase_primitive_data();
                       
                    }
                    
                    if(show_unit_lattice_data || show_lattice_data)
                    {
                        show_lattice_data = false;
                        show_unit_lattice_data = false;
                        erase_lattice_data();
                      

                    }

                    if((show_topo_lattice || topo_done_lattice_do ))
                    {
                    
                        show_topo_lattice = false;
                        topo_done_lattice_do = false;
                        erase_topo_data();
                       
                    }


                    if(ImGui::MenuItem("STRUCTURAL",NULL,&ImguiApp::structural))
                    {
                        ImguiApp::make_inactive(physics_bools,&ImguiApp::structural);
                    }
            
                    else if(ImGui::MenuItem("THERMAL",NULL,&ImguiApp::thermal))
                    {
                        ImguiApp::make_inactive(physics_bools,&ImguiApp::thermal);
                        
                    }

                    ImGui::EndMenu();
                }
                
            
                ImGui::EndMenu();
            }

            if((ImguiApp::structural || ImguiApp::thermal) && ImguiApp::vulkan_buffer_created)
            {
                
                ImguiApp::show_model = false;
                

                if (ImGui::BeginMenu("Selection  "))
                {
                    ImguiApp::make_all_inactive(window_bools);

                    if(ImguiApp::structural)
                    {
                        if (ImGui::BeginMenu("Apply Loads"))
                        {
                            if(ImGui::MenuItem("Select Node ",NULL,&VulkanBaseApp::select_load_node))
                            {
                                ImguiApp::make_inactive(window_bools,&select_load_node);
                            }

                            ImGui::EndMenu();
                        }

                        if (ImGui::BeginMenu("Apply Supports"))
                        {
                            if(ImGui::MenuItem("Select Node ",NULL,&VulkanBaseApp::select_support_node))
                            {
                                ImguiApp::make_inactive(window_bools,&select_support_node);
                            }
                            
                            ImGui::EndMenu();
                        }
                    }

                    if(ImguiApp::thermal)
                    {
                            if (ImGui::BeginMenu("Apply Source"))
                        {
                            if(ImGui::MenuItem("Select Node ",NULL,&VulkanBaseApp::select_load_node))
                            {
                                ImguiApp::make_inactive(window_bools,&select_load_node);
                            }

                            ImGui::EndMenu();
                        }

                        if (ImGui::BeginMenu("Apply Sink"))
                        {
                            if(ImGui::MenuItem("Select Node ",NULL,&VulkanBaseApp::select_support_node))
                            {
                                ImguiApp::make_inactive(window_bools,&select_support_node);
                            }
                            
                            ImGui::EndMenu();
                        }
                    }

                    ImGui::EndMenu();
                }


                if(ImGui::BeginMenu("Analysis "))
                {
                    ImguiApp::make_all_inactive(window_bools);
                    
                    if (ImGui::MenuItem("FEA Settings",NULL,&fea_settings))
                    {
                        ImguiApp::make_inactive(window_bools,&fea_settings);
                    };
                    

                    if (ImGui::BeginMenu("Solver"))
                    {
                        if (ImGui::MenuItem("Conjugate Gradient ",NULL,&cg_solver_settings))
                        {
                            ImguiApp::make_inactive(window_bools,&cg_solver_settings);
                        };

                        ImGui::EndMenu();
                    }

                    ImGui::EndMenu();
                }


                if (ImGui::BeginMenu("Optimisation  "))
                {
                
                    ImguiApp::make_all_inactive(window_bools);

                    if (ImGui::MenuItem("OPTIMISATION Settings",NULL,&optimisation_settings))
                    {
                        ImguiApp::make_inactive(window_bools,&optimisation_settings);
                    };

                    ImGui::EndMenu();
                }

            }
            else if((ImguiApp::structural || ImguiApp::thermal) && (!ImguiApp::vulkan_buffer_created))
            {
                debug_window = true;
                ImguiApp::structural = false;
                ImguiApp::thermal = false;
            }
        

            if(primitives && ImguiApp::boundary_buffers)
            {
                if (ImGui::BeginMenu("Primitives  "))
                {
                    ImguiApp::show_model = true;
                    ImguiApp::make_all_inactive(window_bools);
                    
                    if (ImGui::BeginMenu("Cylinder"))
                    {
                        
                        if(ImGui::MenuItem("Axis & Center",NULL,&cylind_selected))
                        {
                            ImguiApp::make_inactive(window_bools,&cylind_selected);
                        }
                        if(ImGui::MenuItem("Cylinder Disc ",NULL,&cylind_disc_selected))
                        {
                            ImguiApp::make_inactive(window_bools,&cylind_disc_selected);
                        }

                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Cuboid"))
                    {
    
                        if(ImGui::MenuItem("Cuboid Parameters",NULL,&cuboid_selected))
                        {
                            ImguiApp::make_inactive(window_bools,&cuboid_selected);
                        }
                        
                        if(ImGui::MenuItem("Cuboid Shell Parameters",NULL,&cuboid_shell_selected))
                        {
                            ImguiApp::make_inactive(window_bools,&cuboid_shell_selected);
                        }
                        
                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Sphere"))
                    {

                        
                        if(ImGui::MenuItem("Sphere Parameters",NULL,&sphere_selected))
                        {
                            ImguiApp::make_inactive(window_bools,&sphere_selected);
                        }

                        if(ImGui::MenuItem("Sphere Shell Parameters",NULL,&sphere_shell_selected))
                        {
                            ImguiApp::make_inactive(window_bools,&sphere_shell_selected);
                        }
                        
                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("Torus"))
                    {

                        
                        if(ImGui::MenuItem("Torus Parameters",NULL,&torus_selected))
                        {
                            ImguiApp::make_inactive(window_bools,&torus_selected);
                        }
                        
                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("Cone"))
                    {
                        ImguiApp::make_inactive(window_bools,&cone_selected);

                        ImGui::MenuItem("Cone Parameters",NULL,&cone_selected);
                        
                        ImGui::EndMenu();
                    }

                    ImGui::EndMenu();
                }
            }
            else if(primitives && !ImguiApp::boundary_buffers)
            {
                debug_window = true;
                primitives = false;
            }


            
            if((lattice || show_topo_lattice) && lattice_buffer_created)
            {
                if (ImGui::BeginMenu("Lattice  "))
                {
                    ImguiApp::show_model = false;
                    
                    ImguiApp::make_all_inactive(window_bools);
                    
                    if (ImGui::BeginMenu("Unit Lattice  "))
                    {
                        
                        if (ImGui::MenuItem("Settings",NULL,&unit_lattice_settings))
                        {
                            ImguiApp::make_inactive(window_bools,&unit_lattice_settings);
                        };
                        ImGui::EndMenu();
                    }

                    
                    if (ImGui::BeginMenu("Spatial Lattice  "))
                    {
                        if (ImGui::MenuItem("ANGLE",NULL,&spatial_angle_window))
                        {
                            ImguiApp::make_inactive(window_bools,&spatial_angle_window);
                        };

                        if (ImGui::MenuItem("PERIOD",NULL,&spatial_period_window))
                        {
                            ImguiApp::make_inactive(window_bools,&spatial_period_window);
                        }

                        ImGui::EndMenu();

                    }   

                    ImGui::EndMenu();
                }
            }
            else if(lattice && !lattice_buffer_created)
            {
                debug_window = true;
                lattice = false;
            }
            


            if(!ImguiApp::primitives)
            {
                if (ImGui::BeginMenu("Execute  "))
                {
                    ImguiApp::make_all_inactive(window_bools);
                    
            
                    if(ImguiApp::structural || ImguiApp::thermal)
                    {
                        if(ImGui::MenuItem("OPTIMIZE ",NULL,&execute_optimize))
                        {
                    
                            ImguiApp::make_inactive(window_bools,&execute_optimize);
                        }
                    
                    }
                    if(lattice)
                    {
                        if(ImGui::MenuItem("LATTICE ",NULL,&execute_lattice))
                        {
                    
                            ImguiApp::make_inactive(window_bools,&execute_lattice);
                        }
                    
                    }
                    ImGui::EndMenu();
                }
            }
            else 
            {
                if (ImGui::BeginMenu("CLEAR PRIMITIVE "))
                {
                    ImguiApp::make_all_inactive(window_bools);
                
                    if(ImGui::Button("CLEAR PRIMITIVE DATA"))
                    {
                        if(ImguiApp::boundary_buffers)
                        {
                            execute_done = true;
                        }
                        else
                        {
                            ImguiApp::debug_window = true;
                        }
                    }

                    ImGui::EndMenu();
                }
            }
            

            ImGui::EndMainMenuBar();
        }
    

        if(select_load_node)
        {
            if(ImguiApp::structural)
            {
                ImguiApp::show_select_load_structure(&ImguiApp::select_load_node);
                
            }

            else if (ImguiApp::thermal)
            {
                ImguiApp::show_select_load_thermal();
            }

            else
            {
                ImGui::Text("Select a Physics");  
            }

            ImguiApp::make_inactive(window_bools,&select_load_node);
        }


        if(select_support_node)
        {
            if(ImguiApp::structural)
            {
                ImguiApp::show_select_support_structure(&ImguiApp::select_support_node);
                
            }

            else if (ImguiApp::thermal)
            {
                ImguiApp::show_select_support_thermal();
            }

            else
            {
                ImGui::Text("Select a Physics");  
            }
            
        }


        if(grid_settings)
        {
            ImguiApp::show_grid_settings(&grid_settings,vulkan_buffer_created,clear_color);
        }

        if(background_color)
        {
            ImguiApp::show_background_color_settings(&background_color, clear_color);
        }

        if(view_settings)
        {       
            ImguiApp::show_view_settings(&view_settings,&shift,&reset, &show_grid, &show_mesh ,&VulkanBaseApp::push_constants.p_size_1, &VulkanBaseApp::push_constants.p_size_2, &VulkanBaseApp::push_constants.p_size_3,
            &VulkanBaseApp::push_constants.p_size_4);
        }

        if (cylind_selected || cylind_disc_selected)
        {
         
            ImguiApp::show_cylinder_axis_center(
            ImguiApp::center,ImguiApp::angles,ImguiApp::axis,ImguiApp::execute_done,
            ImguiApp::execute_signal,&shift,&reset,&retain,&undoo, &calculate);
            
        }

        if(sphere_selected)
        {
            ImguiApp::show_cylinder_axis_center(
            ImguiApp::center,ImguiApp::angles,ImguiApp::axis, ImguiApp::execute_done,
            ImguiApp::execute_signal,&shift,&reset,&retain,&undoo, &calculate);
        }

        if(sphere_shell_selected)
        {
            ImguiApp::show_cylinder_axis_center(
            ImguiApp::center,ImguiApp::angles,ImguiApp::axis, ImguiApp::execute_done,
            ImguiApp::execute_signal,&shift,&reset,&retain,&undoo, &calculate);
        }

        if(cuboid_selected)
        {
            ImguiApp::show_cylinder_axis_center(
            ImguiApp::center,ImguiApp::angles,ImguiApp::axis,ImguiApp::execute_done,
            ImguiApp::execute_signal,&shift,&reset,&retain,&undoo, &calculate);
        }

        if(cuboid_shell_selected)
        {
            ImguiApp::show_cylinder_axis_center(
            ImguiApp::center,ImguiApp::angles,ImguiApp::axis, ImguiApp::execute_done,
            ImguiApp::execute_signal,&shift,&reset,&retain,&undoo, &calculate);
        }
        if(torus_selected)
        {
            ImguiApp::show_cylinder_axis_center(
            ImguiApp::center,ImguiApp::angles,ImguiApp::axis, ImguiApp::execute_done,
            ImguiApp::execute_signal,&shift,&reset,&retain,&undoo, &calculate);
        }

        if(cone_selected)
        {
            ImguiApp::show_cylinder_axis_center(
            ImguiApp::center,ImguiApp::angles,ImguiApp::axis, ImguiApp::execute_done,
            ImguiApp::execute_signal,&shift,&reset,&retain,&undoo, &calculate);
        }

        if(VulkanBaseApp::undoo)
        {
            erase_primitive_data();
        }

        if(spatial_angle_window)
        {
            show_spatial_angle_settings();
        }

        if(spatial_period_window)
        {
            show_spatial_period_settings();
            
        }

        if(fea_settings)
        {
            show_fea_settings();
        }

        if(cg_solver_settings)
        {
            show_cg_solver_settings();
        }

        if(optimisation_settings)
        {
            show_optimisation_settings();
        }

        if(unit_lattice_settings)
        {
            show_unit_lattice_settings();
        }

        if(execute_optimize)
        {
            show_execute_topo(&ImguiApp::execute_optimize,&ImguiApp::execute_signal,&ImguiApp::execute_done);
        }

        if(execute_lattice)
        {
            show_execute_lattice(&ImguiApp::execute_lattice,&ImguiApp::execute_signal,&ImguiApp::execute_done);
        }

        if(export_settings)
        {
            show_export_settings();
        }

        if(debug_window)
        {
            make_inactive(window_bools,&debug_window);
            show_debugging_window();
        }


   
        ImGui::Render();
        
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),commandBuffer,NULL);
 
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to record command buffer!");
        }
     
   
}



void VulkanBaseApp::createSyncObjects()
{
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image available semaphore!");
        }
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image available semaphore!");
        }
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image available fence!");
        }
        
    }
}

void VulkanBaseApp::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector<VkPipelineStageFlags>& waitStages) const
{
}

void VulkanBaseApp::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{
}

VkDeviceSize VulkanBaseApp::getUniformSize() const
{
    return VkDeviceSize(0);
}

void VulkanBaseApp::updateStorageBuffer(uint32_t imageIndex)
{
}

void VulkanBaseApp::updateUniformBuffer(uint32_t imageIndex, bool shift)
{
}

void VulkanBaseApp::update_inputevents()
{

}

void VulkanBaseApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void VulkanBaseApp::createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;


    VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
    vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    VkExternalMemoryBufferCreateInfo vulkanExternalMemoryBufferCreateInfo ={};
    vulkanExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;

    vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
    vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    vulkanExternalMemoryBufferCreateInfo.pNext = NULL;
    vulkanExternalMemoryBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;


    bufferInfo.pNext = &vulkanExternalMemoryBufferCreateInfo;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create external buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate external buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void *VulkanBaseApp::getMemHandle(VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagBits handleType)
{

    int fd = -1;

    VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
    vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    vkMemoryGetFdInfoKHR.pNext = NULL;
    vkMemoryGetFdInfoKHR.memory = memory;
    vkMemoryGetFdInfoKHR.handleType = handleType;

    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
    fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
  
    if (fpGetMemoryFdKHR(device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }
    return (void *)(uintptr_t)fd;
   
}

void *VulkanBaseApp::getSemaphoreHandle(VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType)
{
   
    int fd;

    VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
    semaphoreGetFdInfoKHR.sType =VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    semaphoreGetFdInfoKHR.pNext = NULL;
    semaphoreGetFdInfoKHR.semaphore = semaphore;
    semaphoreGetFdInfoKHR.handleType = handleType;

    PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
    fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
 
    if (fpGetSemaphoreFdKHR(device, &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }

    return (void *)(uintptr_t)fd;

  
}

void VulkanBaseApp::createExternalSemaphore(VkSemaphore& semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType)
{
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
    exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;


    exportSemaphoreCreateInfo.pNext = NULL;
  
    exportSemaphoreCreateInfo.handleTypes = handleType;
    semaphoreInfo.pNext = &exportSemaphoreCreateInfo;

    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("failed to create synchronization objects for a CUDA-Vulkan!");
    }
}

void VulkanBaseApp::importExternalBuffer(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);


    VkImportMemoryFdInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    handleInfo.pNext = NULL;
    handleInfo.fd = (int)(uintptr_t)handle;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
   

    VkMemoryAllocateInfo memAllocation = {};
    memAllocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocation.pNext = (void *)&handleInfo;
    memAllocation.allocationSize = size;
    memAllocation.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &memAllocation, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to import allocation!");
    }

    vkBindBufferMemory(device, buffer, memory, 0);
}

void VulkanBaseApp::copyBuffer(VkBuffer dst, VkBuffer src, VkDeviceSize srcOffset,VkDeviceSize size)
{

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = srcOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}


void VulkanBaseApp::drawFrame(bool shift)
{
    

    size_t currentFrameIdx = currentFrame % MAX_FRAMES_IN_FLIGHT;

    vkWaitForFences(device, 1, &inFlightFences[currentFrameIdx], VK_TRUE, std::numeric_limits<uint64_t>::max());
    uint32_t imageIndex;

    VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrameIdx], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        
        recreateSwapChain();
      
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }


    updateUniformBuffer(currentFrameIdx,shift);

    update_inputevents();
    ///////////////////////////////////////////////////////////////////////////////////////////////

    updatecommandBuffers(commandBuffers[currentFrameIdx],imageIndex,currentFrameIdx);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStages;
    
    waitSemaphores.push_back(imageAvailableSemaphores[currentFrameIdx]);
    waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    getWaitFrameSemaphores(waitSemaphores, waitStages);

    submitInfo.waitSemaphoreCount = (uint32_t)waitSemaphores.size();
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();


    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrameIdx];


    std::vector<VkSemaphore> signalSemaphores;
    getSignalFrameSemaphores(signalSemaphores);
    signalSemaphores.push_back(renderFinishedSemaphores[currentFrameIdx]);
    submitInfo.signalSemaphoreCount = (uint32_t)signalSemaphores.size();
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    vkResetFences(device, 1, &inFlightFences[currentFrameIdx]);

    VkResult res = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrameIdx]);
    if  (res != VK_SUCCESS) {
        
        throw std::runtime_error("failed to submit draw command buffer!");
    }
    
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrameIdx];

    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        recreateSwapChain();
        framebufferResized = false;
  
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }
    
    currentFrame++;
  
}




void VulkanBaseApp::cleanupSwapChain()
{
      
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        if (depthImageViews[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(device, depthImageViews[i], nullptr);
        }
        if (depthImages[i] != VK_NULL_HANDLE) {
            vkDestroyImage(device, depthImages[i], nullptr);
        }
        if (depthImageMemory[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, depthImageMemory[i], nullptr);
        }
    }

    for (size_t i = 0; i < swapChainImages.size(); i++) 
    {
        if (colorImageViews[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(device, colorImageViews[i], nullptr);
        }
        if (colorImages[i] != VK_NULL_HANDLE) {
            vkDestroyImage(device, colorImages[i], nullptr);
        }
        if (colorImageMemory[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, colorImageMemory[i], nullptr);
        }
    }

    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
        vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
    }


      for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vkDestroyImageView(device, swapChainImageViews[i], nullptr);
    }


      if (swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }
   
    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    if (pipelines.graphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipeline, nullptr);
    }

    if (pipelines.graphicsPipelineread != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipelineread, nullptr);
    }

    if (pipelines.graphicsPipelineone != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipelineone, nullptr);
    }

    if (pipelines.graphicsPipelineoneread != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipelineoneread, nullptr);
    }


    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }


    if (pipelineLayoutread != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayoutread, nullptr);
    }

    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, renderPass, nullptr);
    }

}

void VulkanBaseApp::clean_up()
{
    for (size_t i = 0; i < uniformBuffers.size(); i++) {
    vkDestroyBuffer(device, uniformBuffers[i], nullptr);
    vkFreeMemory(device, uniformMemory[i], nullptr);
    }
    
    for (size_t i = 0; i < storageBuffers.size(); i++) {
        vkDestroyBuffer(device, storageBuffers[i], nullptr);
        vkFreeMemory(device, storageMemory[i], nullptr);
    }



    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    if (descriptorSetLayoutread != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayoutread, nullptr);
    }

}

void VulkanBaseApp::cleanupSyncObjects() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
}

void VulkanBaseApp::recreateSwapChain()
{
    

    int width = 0, height = 0;

    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }
    
    vkDeviceWaitIdle(device);
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createCommandPool();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createCommandBuffers();

}

void VulkanBaseApp::mainLoop(bool shift)

{
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame(shift);
    }
    std::cout<<" Mainloop Terminated "<<std::endl;
    vkDeviceWaitIdle(device);
    std::cout<<"Exiting Appplication"<<std::endl;
}

void readFile(std::istream& s, std::vector<char>& data)
{
    s.seekg(0, std::ios_base::end);
    data.resize(s.tellg());
    s.clear();
    s.seekg(0, std::ios_base::beg);
    s.read(data.data(), data.size());
}





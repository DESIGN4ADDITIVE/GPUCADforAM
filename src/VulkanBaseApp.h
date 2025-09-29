

#pragma once
#ifndef __VULKANBASEAPP_H__
#define __VULKANBASEAPP_H__



#include <string>
#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#include "ImguiApp.h"
#include "Modelling.h"


struct GLFWwindow;

class VulkanBaseApp :public ImguiApp
{
public:
    
    VulkanBaseApp(const std::string& appName, bool enableValidation = false);
    static VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType();
    static VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType();
    virtual ~VulkanBaseApp();
    void init();
    void *getMemHandle(VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagBits handleType);
    void *getSemaphoreHandle(VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);
    void createExternalSemaphore(VkSemaphore& semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void importExternalBuffer(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory);
    void copyBuffer(VkBuffer dst, VkBuffer src, VkDeviceSize srcOffset,VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void mainLoop(bool shift);
   

protected:

    bool topo_data;
    bool show_grid;
    bool show_mesh;
    bool shift;
    bool reset;

    

    static const size_t max_inflight_frames;

    const std::string appName;
    const bool enableValidation;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;


    VkFormat colorFormat;
    VkExtent2D colorExtent;


    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFiles;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFilesread;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFilesone;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFilesoneread;

    VkRenderPass renderPass;

    VkPipelineLayout pipelineLayout;
    VkPipelineLayout pipelineLayoutread;
 
    std::vector<VkFramebuffer> swapChainFramebuffers;
    int vpcount;
    std::vector< VkViewport> viewports ;
    std::vector<VkRect2D> scissors;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkBuffer> storageBuffers;
    std::vector<VkDeviceMemory> storageMemory;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformMemory;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayout descriptorSetLayoutread;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkDescriptorSet> descriptorSetsread;
    std::vector<VkImage> colorImages;
    std::vector<VkDeviceMemory> colorImageMemory;
    std::vector<VkImageView> colorImageViews;

    std::vector<VkImage> depthImages;
    std::vector<VkDeviceMemory> depthImageMemory;
    std::vector<VkImageView> depthImageViews;

    size_t currentFrame;
    bool framebufferResized;
    uint8_t  vkDeviceUUID[VK_UUID_SIZE];
    uint grid_value;
    
    virtual void initVulkanCuda_semaphores() {};
    virtual void fillRenderingCommandBuffer(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBuffer_subpass1(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBufferone_subpass1(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBufferone(VkCommandBuffer& buffer) {};

    virtual void fillRenderingCommandBuffer_unit_lattice(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBuffer_unit_lattice_subpass1(VkCommandBuffer& buffer) {};

    virtual void fillRenderingCommandBuffer_spatial_lattice(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBuffer_spatial_lattice_subpass1(VkCommandBuffer& buffer) {};

    virtual void fillRenderingCommandBuffertwo(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBuffertwo_subpass1(VkCommandBuffer& buffer) {};

    virtual void fillRenderingCommandBufferthree(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBufferthree_subpass1(VkCommandBuffer& buffer) {};

    virtual void fillRenderingCommandBufferfour(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBufferfour_subpass1(VkCommandBuffer& buffer) {};

    virtual std::vector<const char *> getRequiredExtensions() const;
    virtual std::vector<const char *> getRequiredDeviceExtensions() const;
    virtual void getVertexDescriptions(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    virtual void getVertexDescriptionsone(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    virtual void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info);
    virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector< VkPipelineStageFlags>& waitStages) const;
    virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
    virtual VkDeviceSize getUniformSize() const;
    virtual void updateUniformBuffer(uint32_t imageIndex,bool shift);
    virtual void update_inputevents();
    virtual void updateStorageBuffer(uint32_t imageIndex);
    virtual void drawFrame(bool shift);
    virtual void createStorageBuffers(size_t nVerts);

    virtual void erase_topo_data();
    virtual void erase_lattice_data();
    virtual void erase_primitive_data();


    void updatecommandBuffers(VkCommandBuffer commandBuffer, uint32_t image_index, uint32_t currentFrame);
    GLFWwindow *window;

    const GLFWvidmode* videoMode;
    static int screen_width;
    static int screen_height;
 
private:
    
    void initWindow();
    void initVulkan();
    void initImgui();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createDescriptorSetLayoutread();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createColorResources();
    void createDepthResources();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createDescriptorSetsread();
    void createCommandBuffers();
    void createSyncObjects();
    void cleanupSwapChain();
    void clean_up();
    void recreateSwapChain();
    void cleanupSyncObjects();

    static void resizeCallback(GLFWwindow *window, int width, int height);
};

void readFile(std::istream& s, std::vector<char>& data);

#endif /* __VULKANBASEAPP_H__ */
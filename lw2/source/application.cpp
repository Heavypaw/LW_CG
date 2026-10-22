#include "application.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {
namespace {

// ============================================================================
// БЛОК 1: ГЕОМЕТРИЯ
// ============================================================================
// В Лабе 2 нам нужно правильное освещение. Для этого одной позиции вершины 
// недостаточно — нужна нормаль (вектор, перпендикулярный грани).
// У параллелепипеда 8 физических углов, но у каждой грани своя нормаль.
// Если использовать 8 вершин, то одна и та же вершина на стыке граней 
// будет иметь только одну нормаль, и свет ляжет криво (как на сглаженной сфере).
// Поэтому мы генерируем 24 вершины (по 4 на каждую из 6 граней).

struct Vertex {
    float position[3]; // Координаты X, Y, Z
    float normal[3];   // Вектор нормали (направлен наружу от грани)
};

constexpr float W = 1.6f; // Ширина
constexpr float H = 1.0f; // Высота
constexpr float D = 0.7f; // Глубина

std::vector<Vertex> generateVertices() {
    std::vector<Vertex> v;
    const float hw = W * 0.5f, hh = H * 0.5f, hd = D * 0.5f;

    // Лямбда для удобного добавления вершины
    auto add = [&](float x, float y, float z, float nx, float ny, float nz) {
        v.push_back({{x, y, z}, {nx, ny, nz}});
    };

    // Грань 1: Передняя (+Z). Нормаль смотрит на нас (0, 0, 1)
    add(-hw, -hh,  hd, 0, 0, 1);
    add( hw, -hh,  hd, 0, 0, 1);
    add( hw,  hh,  hd, 0, 0, 1);
    add(-hw,  hh,  hd, 0, 0, 1);
    
    // Грань 2: Задняя (-Z). Нормаль смотрит от нас (0, 0, -1)
    add( hw, -hh, -hd, 0, 0, -1);
    add(-hw, -hh, -hd, 0, 0, -1);
    add(-hw,  hh, -hd, 0, 0, -1);
    add( hw,  hh, -hd, 0, 0, -1);
    
    // Грань 3: Правая (+X). Нормаль смотрит вправо (1, 0, 0)
    add( hw, -hh,  hd, 1, 0, 0);
    add( hw, -hh, -hd, 1, 0, 0);
    add( hw,  hh, -hd, 1, 0, 0);
    add( hw,  hh,  hd, 1, 0, 0);
    
    // Грань 4: Левая (-X). Нормаль смотрит влево (-1, 0, 0)
    add(-hw, -hh, -hd, -1, 0, 0);
    add(-hw, -hh,  hd, -1, 0, 0);
    add(-hw,  hh,  hd, -1, 0, 0);
    add(-hw,  hh, -hd, -1, 0, 0);
    
    // Грань 5: Верхняя (+Y). Нормаль смотрит вверх (0, 1, 0)
    add(-hw,  hh,  hd, 0, 1, 0);
    add( hw,  hh,  hd, 0, 1, 0);
    add( hw,  hh, -hd, 0, 1, 0);
    add(-hw,  hh, -hd, 0, 1, 0);
    
    // Грань 6: Нижняя (-Y). Нормаль смотрит вниз (0, -1, 0)
    add(-hw, -hh, -hd, 0, -1, 0);
    add( hw, -hh, -hd, 0, -1, 0);
    add( hw, -hh,  hd, 0, -1, 0);
    add(-hw, -hh,  hd, 0, -1, 0);
    
    return v;
}

std::vector<uint32_t> generateIndices() {
    std::vector<uint32_t> idx;
    for (uint32_t face = 0; face < 6; ++face) {
        uint32_t b = face * 4;
        // На каждую грань по 2 треугольника (6 индексов)
        idx.push_back(b + 0); idx.push_back(b + 1); idx.push_back(b + 2);
        idx.push_back(b + 0); idx.push_back(b + 2); idx.push_back(b + 3);
    }
    return idx;
}

// Глобальные векторы с геометрией
std::vector<Vertex>   vertices = generateVertices();
std::vector<uint32_t> indices  = generateIndices();

// ============================================================================
// БЛОК 2: СТРУКТУРЫ ДАННЫХ (Мост между C++ и Шейдерами)
// ============================================================================

// --- Структуры для CPU (хранение данных и ImGUI) ---
struct PointLightCPU {
    glm::vec3 position  = {  2.0f, 2.0f,  2.0f };
    glm::vec3 color     = {  1.0f, 1.0f,  1.0f };
    float     intensity = 10.0f;
    float     radius    = 10.0f;
};

struct SpotLightCPU {
    glm::vec3 position  = {  0.0f, 3.0f,  2.0f };
    glm::vec3 direction = {  0.0f, -1.0f, -1.0f };
    glm::vec3 color     = {  1.0f, 1.0f,  1.0f };
    float     intensity = 15.0f;
    float     radius    = 15.0f;
    float     inner_deg = 15.0f;
    float     outer_deg = 30.0f;
};

// --- Структуры для GPU (должны совпадать с layout в шейдерах) ---
// alignas(16) гарантирует, что данные будут выровнены в памяти так, как 
// этого требует видеокарта (стандарт std140 для uniform буферов).

struct alignas(16) SceneUniforms {
    glm::mat4 view_projection;       // Матрица камеры (Projection * View)
    glm::vec4 camera_position;       // Позиция камеры (нужна для расчета бликов)
    glm::vec4 ambient_color;         // Фоновый цвет
    glm::vec4 directional_direction; // Куда светит направленный свет
    glm::vec4 directional_color;     // RGB цвет + W интенсивность
    glm::vec4 counts;                // X = кол-во Point Lights, Y = кол-во Spot Lights
};

struct alignas(16) ModelUniforms {
    glm::mat4 model;            // Матрица трансформации объекта
    glm::vec4 albedo;           // Основной цвет материала
    glm::vec4 specular_color;   // RGB цвет блика + W параметр Shininess
};

struct alignas(16) PointLightGPU {
    glm::vec4 position;         // XYZ позиция
    glm::vec4 color;            // RGB цвет + W интенсивность
    glm::vec4 params;           // X = радиус действия
};

struct alignas(16) SpotLightGPU {
    glm::vec4 position;         // XYZ позиция
    glm::vec4 direction;        // XYZ направление
    glm::vec4 color;            // RGB цвет + W интенсивность
    glm::vec4 params;           // X = радиус, Y = cos(inner), Z = cos(outer)
};

constexpr size_t MAX_LIGHTS = 32; // Ограничение размера Storage буфера

// ============================================================================
// БЛОК 3: СОСТОЯНИЕ UI (Изменяется ползунками в ImGUI)
// ============================================================================
// Трансформации объекта (Лаба 1)
glm::vec3 ui_position = { 0.0f, 0.0f, 0.0f };
glm::vec3 ui_rotation = { 0.0f, 0.0f, 0.0f };
glm::vec3 ui_scale    = { 1.0f, 1.0f, 1.0f };
int       ui_use_perspective = 1;

// Камера
glm::vec3 ui_camera_position = { 0.0f, 2.0f, 5.0f }; // Где стоит камера
glm::vec3 ui_camera_target   = { 0.0f, 0.0f, 0.0f }; // Куда она смотрит
float     ui_camera_fov      = 45.0f;                // Угол обзора

// Свет и материал
glm::vec3 ui_ambient_color = { 0.1f, 0.1f, 0.1f };
glm::vec3 ui_dir_direction = { -1.0f, -1.0f, -1.0f };
glm::vec3 ui_dir_color     = {  1.0f,  1.0f,  1.0f };
float     ui_dir_intensity = 1.0f;
glm::vec3 ui_albedo         = { 0.8f, 0.8f, 0.8f };
glm::vec3 ui_specular_color = { 1.0f, 1.0f, 1.0f };
float     ui_shininess      = 32.0f;

std::vector<PointLightCPU> point_lights;
std::vector<SpotLightCPU>  spot_lights;

// ============================================================================
// БЛОК 4: VULKAN ОБЪЕКТЫ (Создаются один раз при старте)
// ============================================================================
VkBuffer       vk_vertex_buffer          = VK_NULL_HANDLE;
VmaAllocation  vma_vertex_buffer_alloc   = VK_NULL_HANDLE;
VkBuffer       vk_index_buffer           = VK_NULL_HANDLE;
VmaAllocation  vma_index_buffer_alloc    = VK_NULL_HANDLE;

VkBuffer       vk_scene_uniform_buffer   = VK_NULL_HANDLE;
VmaAllocation  vma_scene_uniform_alloc   = VK_NULL_HANDLE;
SceneUniforms* scene_uniform_mapped      = nullptr;

VkBuffer       vk_model_uniform_buffer   = VK_NULL_HANDLE;
VmaAllocation  vma_model_uniform_alloc   = VK_NULL_HANDLE;
ModelUniforms* model_uniform_mapped      = nullptr;

VkBuffer        vk_point_storage_buffer  = VK_NULL_HANDLE;
VmaAllocation   vma_point_storage_alloc  = VK_NULL_HANDLE;
PointLightGPU*  point_storage_mapped     = nullptr;

VkBuffer        vk_spot_storage_buffer   = VK_NULL_HANDLE;
VmaAllocation   vma_spot_storage_alloc   = VK_NULL_HANDLE;
SpotLightGPU*   spot_storage_mapped      = nullptr;

// Два набора дескрипторов (Доп. 1)
VkDescriptorSetLayout vk_scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout vk_model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool      vk_descriptor_pool  = VK_NULL_HANDLE;
VkDescriptorSet       vk_scene_descriptor = VK_NULL_HANDLE;
VkDescriptorSet       vk_model_descriptor = VK_NULL_HANDLE;

VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkPipeline       vk_pipeline        = VK_NULL_HANDLE;

// ============================================================================
// БЛОК 5: ХЕЛПЕРЫ И СОЗДАНИЕ РЕСУРСОВ
// ============================================================================
VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    file.close();

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode    = reinterpret_cast<const uint32_t*>(buf.data());

    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &m) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return m;
}

// Создание всех буферов (Вершины, Индексы, Uniform, Storage)
bool createBuffers() {
    auto& ctx = graphics::internal::context;

    // Лямбда: создает буфер и сразу копирует туда данные (для вершин и индексов)
    auto create_and_copy = [&](size_t size, VkBufferUsageFlags usage, const void* data,
                               VkBuffer* out_buf, VmaAllocation* out_alloc) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        alloc.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo vma_info{};
        if (vmaCreateBuffer(ctx.allocator, &info, &alloc, out_buf, out_alloc, &vma_info) != VK_SUCCESS) return false;
        if (data && vma_info.pMappedData) {
            std::memcpy(vma_info.pMappedData, data, size);
        }
        return true;
    };

    // Лямбда: создает пустой буфер и отдает указатель для записи каждый кадр
    auto create_mapped = [&](size_t size, VkBufferUsageFlags usage,
                             VkBuffer* out_buf, VmaAllocation* out_alloc, void** out_mapped) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        alloc.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo vma_info{};
        if (vmaCreateBuffer(ctx.allocator, &info, &alloc, out_buf, out_alloc, &vma_info) != VK_SUCCESS) return false;
        *out_mapped = vma_info.pMappedData;
        return true;
    };

    // Создаем буферы
    if (!create_and_copy(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data(), &vk_vertex_buffer, &vma_vertex_buffer_alloc)) return false;
    if (!create_and_copy(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data(), &vk_index_buffer, &vma_index_buffer_alloc)) return false;
    if (!create_mapped(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &vk_scene_uniform_buffer, &vma_scene_uniform_alloc, (void**)&scene_uniform_mapped)) return false;
    if (!create_mapped(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &vk_model_uniform_buffer, &vma_model_uniform_alloc, (void**)&model_uniform_mapped)) return false;
    if (!create_mapped(MAX_LIGHTS * sizeof(PointLightGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vk_point_storage_buffer, &vma_point_storage_alloc, (void**)&point_storage_mapped)) return false;
    if (!create_mapped(MAX_LIGHTS * sizeof(SpotLightGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vk_spot_storage_buffer, &vma_spot_storage_alloc, (void**)&spot_storage_mapped)) return false;

    return true;
}

bool createDescriptors() {
    auto& ctx = graphics::internal::context;
//Дескрипторы
    // --- Доп. 1: Создание Scene Set (Set 0) ---
    {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // Scene Uniforms
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Point Lights
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Spot Lights
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 3;
        info.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(ctx.device, &info, nullptr, &vk_scene_set_layout) != VK_SUCCESS) return false;
    }

    // --- Доп. 1: Создание Model Set (Set 1) ---
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // Model Uniforms
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 1;
        info.pBindings = &b;
        if (vkCreateDescriptorSetLayout(ctx.device, &info, nullptr, &vk_model_set_layout) != VK_SUCCESS) return false;
    }

    // Пул дескрипторов
    {
        VkDescriptorPoolSize sizes[2] = {};
        sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = 2;
        sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets = 2;
        info.poolSizeCount = 2;
        info.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(ctx.device, &info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) return false;
    }

    // Выделяем Scene Set и связываем с буферами
    {
        VkDescriptorSetAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = vk_descriptor_pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &vk_scene_set_layout;
        if (vkAllocateDescriptorSets(ctx.device, &info, &vk_scene_descriptor) != VK_SUCCESS) return false;

        VkDescriptorBufferInfo buf_scene{ vk_scene_uniform_buffer, 0, sizeof(SceneUniforms) };
        VkDescriptorBufferInfo buf_pts  { vk_point_storage_buffer, 0, MAX_LIGHTS * sizeof(PointLightGPU) };
        VkDescriptorBufferInfo buf_spt  { vk_spot_storage_buffer,  0, MAX_LIGHTS * sizeof(SpotLightGPU) };

        VkWriteDescriptorSet w[3] = {};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = vk_scene_descriptor;
        w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &buf_scene;
        
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[1].dstSet = vk_scene_descriptor;
        w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &buf_pts;
        
        w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[2].dstSet = vk_scene_descriptor;
        w[2].dstBinding = 2; w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &buf_spt;

        vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);
    }

    // Выделяем Model Set и связываем с буфером
    {
        VkDescriptorSetAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = vk_descriptor_pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &vk_model_set_layout;
        if (vkAllocateDescriptorSets(ctx.device, &info, &vk_model_descriptor) != VK_SUCCESS) return false;

        VkDescriptorBufferInfo buf{ vk_model_uniform_buffer, 0, sizeof(ModelUniforms) };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = vk_model_descriptor;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &buf;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    return true;
}

bool createPipeline() {
    auto& ctx = graphics::internal::context;

    VkShaderModule vert = loadShaderModule(SHADER_DIR "/lab2.vert.spv");
    VkShaderModule frag = loadShaderModule(SHADER_DIR "/lab2.frag.spv");
    if (!vert || !frag) return false;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vert; stages[0].pName = "main";
    
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag; stages[1].pName = "main";

    // Описание вершин: позиция (loc 0) и нормаль (loc 1)
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0; binding.stride = sizeof(Vertex); binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attrs = {};
    attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = offsetof(Vertex, position);
    attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vin{};
    vin.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vin.vertexBindingDescriptionCount = 1; vin.pVertexBindingDescriptions = &binding;
    vin.vertexAttributeDescriptionCount = (uint32_t)attrs.size(); vin.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE; // Отсечение граней отключено, чтобы избежать багов
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cb_att{};
    cb_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cb_att.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; cb.attachmentCount = 1; cb.pAttachments = &cb_att;

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; dyn.dynamicStateCount = 2; dyn.pDynamicStates = dyn_states;

    // Передаем ДВА набора в Pipeline Layout
    VkDescriptorSetLayout set_layouts[2] = { vk_scene_set_layout, vk_model_set_layout };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; pli.setLayoutCount = 2; pli.pSetLayouts = set_layouts;

    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &vk_pipeline_layout) != VK_SUCCESS) return false;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = 2; pi.pStages = stages; pi.pVertexInputState = &vin; pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp; pi.pRasterizationState = &rast; pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds; pi.pColorBlendState = &cb; pi.pDynamicState = &dyn;
    pi.layout = vk_pipeline_layout; pi.renderPass = ctx.render_pass; pi.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pi, nullptr, &vk_pipeline) != VK_SUCCESS) return false;

    vkDestroyShaderModule(ctx.device, vert, nullptr); vkDestroyShaderModule(ctx.device, frag, nullptr);
    return true;
}

} // namespace

// ============================================================================
// БЛОК 6: ПУБЛИЧНЫЕ ФУНКЦИИ
// ============================================================================
bool initialize() {
    if (!createBuffers())     return false;
    if (!createDescriptors()) return false;
    if (!createPipeline())    return false;
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    vkDestroyPipeline(ctx.device, vk_pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);
    vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_scene_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_model_set_layout, nullptr);

    vmaDestroyBuffer(ctx.allocator, vk_spot_storage_buffer,  vma_spot_storage_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_point_storage_buffer, vma_point_storage_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_model_uniform_buffer, vma_model_uniform_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_scene_uniform_buffer, vma_scene_uniform_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_index_buffer,         vma_index_buffer_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer,        vma_vertex_buffer_alloc);
}

void update([[maybe_unused]] double time) {
    auto& ctx = graphics::internal::context;

    // ===================== 1. ИНТЕРФЕЙС IMGUI =====================
    ImGui::Begin("Lab 2 - Camera and Lighting");

    // --- Блок Камеры ---
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Camera");
        // Позиция камеры (Eye) - точка, откуда мы смотрим
        ImGui::DragFloat3("Position", &ui_camera_position.x, 0.05f);
        // Точка, на которую смотрит камера (Target). По умолчанию (0,0,0) - центр объекта
        ImGui::DragFloat3("Target",   &ui_camera_target.x,   0.05f);
        // Угол обзора (Field of View). Чем больше, тем шире видим, но сильнее искажение
        ImGui::SliderFloat("FOV", &ui_camera_fov, 15.0f, 120.0f);
        if (ImGui::Button("Reset Camera")) {
            ui_camera_position = { 0.0f, 2.0f, 5.0f };
            ui_camera_target   = { 0.0f, 0.0f, 0.0f };
            ui_camera_fov      = 45.0f;
        }
        ImGui::PopID();
    }

    // --- Блок Трансформаций объекта ---
    if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Transform");
        ImGui::Text("Projection");
        ImGui::RadioButton("Perspective",  &ui_use_perspective, 1); ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &ui_use_perspective, 0);
        // Позиция самого объекта в мире
        ImGui::SliderFloat3("Position", &ui_position.x, -5.0f, 5.0f);
        // Поворот объекта в градусах по осям X, Y, Z
        ImGui::SliderFloat3("Rotation", &ui_rotation.x, -180.0f, 180.0f);
        // Масштаб (растяжение) по осям
        ImGui::SliderFloat3("Scale",    &ui_scale.x,    0.1f, 3.0f);
        if (ImGui::Button("Reset Transform")) {
            ui_position = { 0.0f, 0.0f, 0.0f };
            ui_rotation = { 0.0f, 0.0f, 0.0f };
            ui_scale    = { 1.0f, 1.0f, 1.0f };
            ui_use_perspective = 1;
        }
        ImGui::PopID();
    }

    // --- Блок Окружения (Ambient) ---
    if (ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Environment");
        ImGui::ColorEdit3("Ambient", &ui_ambient_color.x);
        ImGui::PopID();
    }

    // --- Блок Направленного света (Directional) ---
    if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("DirLight");
        // Вектор направления света (например, (-1,-1,-1) значит светит сверху-слева-спереди)
        ImGui::DragFloat3("Direction", &ui_dir_direction.x, 0.05f);
        ImGui::ColorEdit3("Color", &ui_dir_color.x);
        ImGui::SliderFloat("Intensity", &ui_dir_intensity, 0.0f, 5.0f);
        ImGui::PopID();
    }

    // --- Блок Материала (Блинн-Фонг) ---
    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Material");
        // Основной цвет материала
        ImGui::ColorEdit3("Albedo", &ui_albedo.x);
        // Цвет блика
        ImGui::ColorEdit3("Specular", &ui_specular_color.x);
        // Шероховатость: меньше = более матовый, больше = более глянцевый (блик сжимается)
        ImGui::SliderFloat("Shininess", &ui_shininess, 1.0f, 256.0f);
        ImGui::PopID();
    }

    // --- Блок Точечных источников (Доп. 2) ---
    if (ImGui::CollapsingHeader("Point Lights")) {
        ImGui::Text("Count: %zu", point_lights.size());
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID((int)i);
            ImGui::Separator();
            ImGui::Text("Point #%zu", i);
            ImGui::DragFloat3("Position",  &point_lights[i].position.x, 0.05f);
            ImGui::ColorEdit3("Color",     &point_lights[i].color.x);
            ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 50.0f);
            ImGui::SliderFloat("Radius",   &point_lights[i].radius, 0.5f, 30.0f);
            if (ImGui::Button("Remove")) {
                point_lights.erase(point_lights.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        if (point_lights.size() < MAX_LIGHTS) {
            if (ImGui::Button("Add Point Light")) {
                point_lights.push_back(PointLightCPU{});
            }
        }
    }

    // --- Блок Прожекторов (Доп. 3) ---
    if (ImGui::CollapsingHeader("Spot Lights")) {
        ImGui::Text("Count: %zu", spot_lights.size());
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID((int)(i + 1000));
            ImGui::Separator();
            ImGui::Text("Spot #%zu", i);
            ImGui::DragFloat3("Position",  &spot_lights[i].position.x, 0.05f);
            ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.05f);
            ImGui::ColorEdit3("Color",     &spot_lights[i].color.x);
            ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 100.0f);
            ImGui::SliderFloat("Radius",    &spot_lights[i].radius, 0.5f, 50.0f);
            // Углы для мягких краев
            ImGui::SliderFloat("Inner Angle", &spot_lights[i].inner_deg, 1.0f, 89.0f);
            ImGui::SliderFloat("Outer Angle", &spot_lights[i].outer_deg, 1.0f, 89.0f);
            if (ImGui::Button("Remove")) {
                spot_lights.erase(spot_lights.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        if (spot_lights.size() < MAX_LIGHTS) {
            if (ImGui::Button("Add Spot Light")) {
                spot_lights.push_back(SpotLightCPU{});
            }
        }
    }
    ImGui::End();

    // ===================== 2. МАТЕМАТИКА (КАМЕРА И ТРАНСФОРМАЦИИ) =====================
    float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;

    // --- МАТРИЦА VIEW (КАМЕРА) ---
    // Функция glm::lookAt создает матрицу вида (View).
    // Она берет три вектора:
    // 1. eye (ui_camera_position) - физическая позиция камеры в 3D-пространстве.
    // 2. center (ui_camera_target) - точка, на которую камера направлена (куда смотрит объектив).
    // 3. up (0, 1, 0) - вектор, указывающий, где находится "верх" камеры. 
    //    Без него камера могла бы быть перевернутой или наклоненной в сторону.
    // Эта матрица переводит мировые координаты в координаты относительно камеры.
    glm::mat4 view = glm::lookAt(ui_camera_position, ui_camera_target, glm::vec3(0, 1, 0));

    // --- МАТРИЦА PROJECTION (ПРОЕКЦИЯ) ---
    // Превращает 3D координаты камеры в 2D координаты экрана (клип-пространство).
    glm::mat4 proj;
    if (ui_use_perspective == 1) {
        // Перспективная: дальние объекты меньше. FOV (угол обзора), aspect (соотношение сторон), near/far (плоскости отсечения).
        proj = glm::perspective(glm::radians(ui_camera_fov), aspect, 0.1f, 100.0f);
    } else {
        // Ортографическая: объекты не уменьшаются с расстоянием (как в инженерных чертежах).
        const float half_size = 2.0f;
        proj = glm::ortho(-half_size * aspect, half_size * aspect,
                          -half_size, half_size, 0.1f, 100.0f);
    }

    // --- КОРРЕКЦИЯ ДЛЯ VULKAN ---
    // В OpenGL Z-координата находится в диапазоне от -1 до 1.
    // В Vulkan Z-координата должна быть от 0 до 1.
    // Эта матрица сдвигает и масштабирует Z, чтобы соответствовать стандарту Vulkan.
    glm::mat4 vulkan_clip(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.0f, 0.5f, 1.0f
    );
    proj = vulkan_clip * proj;

    // Итоговая матрица для вершинного шейдера
    glm::mat4 view_projection = proj * view;

    // --- МАТРИЦА MODEL (ТРАНСФОРМАЦИИ ОБЪЕКТА) ---
    // Порядок применения важен: сначала масштабируем (Scale), затем поворачиваем (Rotate),
    // затем перемещаем (Translate). В glm матрицы перемножаются справа налево.
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, ui_position); // Сдвиг
    model = glm::rotate(model, glm::radians(ui_rotation.x), glm::vec3(1, 0, 0)); // Вращение по X
    model = glm::rotate(model, glm::radians(ui_rotation.y), glm::vec3(0, 1, 0)); // Вращение по Y
    model = glm::rotate(model, glm::radians(ui_rotation.z), glm::vec3(0, 0, 1)); // Вращение по Z
    model = glm::scale(model, ui_scale); // Растяжение

    // ===================== 3. ОТПРАВКА ДАННЫХ В GPU =====================
    
    // --- Отправляем Scene Uniforms (Камера и базовый свет) ---
    if (scene_uniform_mapped) {
        scene_uniform_mapped->view_projection       = view_projection;
        scene_uniform_mapped->camera_position       = glm::vec4(ui_camera_position, 1.0f); // Нужна для блика
        scene_uniform_mapped->ambient_color         = glm::vec4(ui_ambient_color, 0.0f);
        scene_uniform_mapped->directional_direction = glm::vec4(glm::normalize(ui_dir_direction), 0.0f);
        scene_uniform_mapped->directional_color     = glm::vec4(ui_dir_color, ui_dir_intensity);
        // Передаем количество активных источников для циклов в шейдере
        scene_uniform_mapped->counts                = glm::vec4((float)point_lights.size(), (float)spot_lights.size(), 0.0f, 0.0f);
    }

    // --- Отправляем Model Uniforms (Материал и матрица объекта) ---
    if (model_uniform_mapped) {
        model_uniform_mapped->model          = model;
        model_uniform_mapped->albedo         = glm::vec4(ui_albedo, 0.0f);
        // В W-компоненту записываем Shininess, чтобы не делать отдельный float
        model_uniform_mapped->specular_color = glm::vec4(ui_specular_color, ui_shininess);
    }

    // --- Отправляем данные Точечных источников ---
    if (point_storage_mapped) {
        for (size_t i = 0; i < point_lights.size(); ++i) {
            auto& L = point_lights[i];
            point_storage_mapped[i].position = glm::vec4(L.position, 0.0f);
            point_storage_mapped[i].color    = glm::vec4(L.color, L.intensity);
            point_storage_mapped[i].params   = glm::vec4(L.radius, 0.0f, 0.0f, 0.0f);
        }
    }

    // --- Отправляем данные Прожекторов ---
    if (spot_storage_mapped) {
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            auto& L = spot_lights[i];
            float inner_rad = glm::radians(L.inner_deg);
            float outer_rad = glm::radians(L.outer_deg);
            spot_storage_mapped[i].position  = glm::vec4(L.position, 0.0f);
            spot_storage_mapped[i].direction = glm::vec4(glm::normalize(L.direction), 0.0f);
            spot_storage_mapped[i].color     = glm::vec4(L.color, L.intensity);
            // Передаем косинусы углов, чтобы шейдеру не нужно было считать их каждый раз
            spot_storage_mapped[i].params    = glm::vec4(L.radius, std::cos(inner_rad), std::cos(outer_rad), 0.0f);
        }
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    vkResetCommandBuffer(fd.command_buffer, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(fd.command_buffer, &begin);

    VkClearValue clears[2] = {};
    clears[0].color = {{ 0.05f, 0.05f, 0.05f, 1.0f }}; // Темный фон
    clears[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = ctx.render_pass;
    rp.framebuffer = fd.framebuffer;
    rp.renderArea.extent = ctx.swapchain_extent;
    rp.clearValueCount = 2;
    rp.pClearValues = clears;

    vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width  = (float)ctx.swapchain_extent.width;
    vp.height = (float)ctx.swapchain_extent.height;
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);

    VkRect2D sc{};
    sc.extent = ctx.swapchain_extent;
    vkCmdSetScissor(fd.command_buffer, 0, 1, &sc);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

    // Привязываем ДВА набора: Scene (0) и Model (1)
    VkDescriptorSet sets[2] = { vk_scene_descriptor, vk_model_descriptor };
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk_pipeline_layout, 0, 2, sets, 0, nullptr);

    // Рисуем объект (24 вершины = 36 индексов)
    vkCmdDrawIndexed(fd.command_buffer, (uint32_t)indices.size(), 1, 0, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
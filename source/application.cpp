#include "application.hpp"

#include <array>      // std::array — массив фиксированного размера
#include <cstring>    // std::memcpy — копирование байтов памяти
#include <fstream>    // чтение файлов с диска (для шейдеров)
#include <iostream>   // std::cerr — вывод ошибок в консоль
#include <vector>     // динамический массив

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <imgui.h>                        // ImGUI — интерфейс
#include <glm/glm.hpp>                    // glm — типы vec3, vec4, mat4
#include <glm/gtc/matrix_transform.hpp>   // glm — translate, rotate, scale, lookAt, perspective, ortho

namespace application {   

namespace {              

// Структура описывает, как выглядит одна вершина в памяти.
// GPU будет читать данные из вершинного буфера, зная этот формат.
struct Vertex {
    float position[3]; // x, y, z — координаты вершины в локальной системе объекта
    float color[3];    // r, g, b — цвет вершины (0.0 - 1.0)
};                     // итого 6 * 4 = 24 байта на вершину

// Разные стороны — это именно параллелепипед, а не куб.
// Все стороны одинаковые = куб (вариант 1). Разные = параллелепипед (вариант 3).
const float W = 1.6f; // ширина  по оси X
const float H = 1.0f; // высота  по оси Y
const float D = 0.7f; // глубина по оси Z

// Центр фигуры находится в начале координат (0,0,0)
const Vertex vertices[] = {
    // Задняя грань (z = -D/2). Смотрим на неё со стороны положительного Z.
    {{-W/2, -H/2, -D/2}, {1.0f, 0.0f, 0.0f}}, // 0 — левый нижний,  красный
    {{ W/2, -H/2, -D/2}, {0.0f, 1.0f, 0.0f}}, // 1 — правый нижний, зелёный
    {{ W/2,  H/2, -D/2}, {0.0f, 0.0f, 1.0f}}, // 2 — правый верхний, синий
    {{-W/2,  H/2, -D/2}, {1.0f, 1.0f, 0.0f}}, // 3 — левый верхний,  жёлтый

    // Передняя грань (z = +D/2)
    {{-W/2, -H/2,  D/2}, {1.0f, 0.0f, 1.0f}}, // 4 — левый нижний,  пурпурный
    {{ W/2, -H/2,  D/2}, {0.0f, 1.0f, 1.0f}}, // 5 — правый нижний, голубой
    {{ W/2,  H/2,  D/2}, {1.0f, 1.0f, 1.0f}}, // 6 — правый верхний, белый
    {{-W/2,  H/2,  D/2}, {0.0f, 0.0f, 0.0f}}, // 7 — левый верхний,  чёрный
};

// Индексы — это номера вершин из массива vertices[].
// GPU читает их парами по три и собирает треугольники.
// У параллелепипеда 6 граней × 2 треугольника × 3 вершины = 36 индексов.
// Благодаря индексам одна вершина используется в нескольких гранях —
// не нужно дублировать её в памяти.
const uint32_t indices[] = {
    0, 1, 2,  2, 3, 0,  // задняя грань   (2 треугольника)
    4, 5, 6,  6, 7, 4,  // передняя грань
    0, 1, 5,  5, 4, 0,  // нижняя грань
    3, 2, 6,  6, 7, 3,  // верхняя грань
    0, 3, 7,  7, 4, 0,  // левая грань
    1, 2, 6,  6, 5, 1,  // правая грань
};

// Через uniform-буфер передаём данные, одинаковые для всего вызова отрисовки.
// Здесь — одна матрица 4x4 (Model-View-Projection).
struct GlobalUniforms {
    glm::mat4 mvp; // 16 float = 64 байта
};

// Эти переменные обновляются из UI-панели. Потом update() читает их
// и строит на их основе MVP-матрицу.
glm::vec3 ui_position = { 0.0f, 0.0f, 0.0f };         // смещение фигуры
glm::vec3 ui_rotation = { 0.0f, 0.0f, 0.0f };         // поворот в градусах
glm::vec3 ui_scale    = { 1.0f, 1.0f, 1.0f };         // масштаб
int       ui_use_perspective = 1;                     // 1 = перспектива, 0 = ортография
float     ui_fov = 45.0f;                             // угол обзора камеры

// --- Вершинный буфер: хранит массив Vertex в памяти GPU ---
VkBuffer        vk_vertex_buffer             = VK_NULL_HANDLE;
VmaAllocation   vma_vertex_buffer_allocation = VK_NULL_HANDLE;

// --- Индексный буфер: хранит массив uint32_t в памяти GPU ---
VkBuffer        vk_index_buffer              = VK_NULL_HANDLE;
VmaAllocation   vma_index_buffer_allocation  = VK_NULL_HANDLE;

// --- Uniform-буфер: хранит GlobalUniforms (MVP-матрицу) ---
VkBuffer        vk_uniform_buffer            = VK_NULL_HANDLE;
VmaAllocation   vma_uniform_buffer_allocation = VK_NULL_HANDLE;
GlobalUniforms* uniform_buffer_mapped        = nullptr; // CPU-указатель на память GPU

// --- Дескрипторы: механизм связывания ресурсов с шейдером ---
VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE; // "схема" набора
VkDescriptorPool      vk_descriptor_pool       = VK_NULL_HANDLE; // "фабрика" наборов
VkDescriptorSet       vk_descriptor_set        = VK_NULL_HANDLE; // конкретный набор

// --- Pipeline: весь конвейер отрисовки ---
VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE; // описание наборов дескрипторов
VkPipeline       vk_pipeline        = VK_NULL_HANDLE; // сам графический pipeline

// Скомпилированный шейдер (SPIR-V) лежит на диске как бинарный файл.
// Эта функция читает его и превращает в объект VkShaderModule.
VkShaderModule loadShaderModule(const char* path) {
    // Открываем файл в бинарном режиме, сразу переходим в конец (ate),
    // чтобы узнать размер файла через tellg().
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open shader file: " << path << '\n';
        return VK_NULL_HANDLE;
    }

    const size_t size = static_cast<size_t>(file.tellg()); // размер файла в байтах
    std::vector<char> buffer(size);                        // буфер под данные
    file.seekg(0);                                         // возвращаемся в начало
    file.read(buffer.data(), size);                        // читаем весь файл
    file.close();

    // Описываем будущий шейдерный модуль для Vulkan.
    // pCode ожидает uint32_t*, поэтому делаем reinterpret_cast.
    // SPIR-V всегда кратен 4 байтам, так что это безопасно.
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode    = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

// Все три буфера создаются через VMA, который сам подбирает тип памяти и управляет фрагментацией.
bool createBuffers() {
    auto& ctx = graphics::internal::context; // сокращение для удобства

    // ---------- ВЕРШИННЫЙ БУФЕР ----------
    {
        // Описание: что за буфер мы хотим создать.
        VkBufferCreateInfo info{};
        info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size        = sizeof(vertices);                       // размер данных
        info.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;      // используется как вершинный
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;              // один поток/очередь

        // Параметры выделения памяти через VMA.
        // MAPPED — сразу замапить (получить CPU-указатель).
        // HOST_ACCESS_SEQUENTIAL_WRITE — писать с CPU последовательно.
        // AUTO — VMA сама подберёт подходящий тип.
        VmaAllocationCreateInfo alloc{};
        alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        alloc.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo alloc_info{};
        if (vmaCreateBuffer(ctx.allocator, &info, &alloc,
                            &vk_vertex_buffer, &vma_vertex_buffer_allocation,
                            &alloc_info) != VK_SUCCESS) {
            std::cerr << "Failed to create vertex buffer\n";
            return false;
        }

        // Копируем данные вершин из массива vertices[] в память GPU.
        // pMappedData — CPU-указатель на ту же память, что видит GPU.
        std::memcpy(alloc_info.pMappedData, vertices, sizeof(vertices));
    }

    // ---------- ИНДЕКСНЫЙ БУФЕР ----------
    {
        VkBufferCreateInfo info{};
        info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size        = sizeof(indices);
        info.usage       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;  // используется как индексный
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        alloc.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo alloc_info{};
        if (vmaCreateBuffer(ctx.allocator, &info, &alloc,
                            &vk_index_buffer, &vma_index_buffer_allocation,
                            &alloc_info) != VK_SUCCESS) {
            std::cerr << "Failed to create index buffer\n";
            return false;
        }

        std::memcpy(alloc_info.pMappedData, indices, sizeof(indices));
    }

    // ---------- UNIFORM-БУФЕР ----------
    {
        VkBufferCreateInfo info{};
        info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size        = sizeof(GlobalUniforms);
        info.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;  // используется как uniform
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        alloc.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo alloc_info{};
        if (vmaCreateBuffer(ctx.allocator, &info, &alloc,
                            &vk_uniform_buffer, &vma_uniform_buffer_allocation,
                            &alloc_info) != VK_SUCCESS) {
            std::cerr << "Failed to create uniform buffer\n";
            return false;
        }

        // Здесь memcpy НЕ делаем: матрица будет писаться каждый кадр в update().
        // Сохраняем указатель, чтобы потом писать через него.
        uniform_buffer_mapped = reinterpret_cast<GlobalUniforms*>(alloc_info.pMappedData);
    }

    return true;
}


// Дескриптор — это "ссылка" шейдера на внешний ресурс
//  Чтобы шейдер смог прочитать binding=0, нужно:
//  1. Описать "схему" (SetLayout) — что за ресурсы будут в наборе.
//  2. Создать пул (Pool) — из него выделяются дескрипторы.
//  3. Выделить конкретный набор (Set) по схеме.
//  4. Привязать к нему uniform-буфер.
bool createDescriptors() {
    auto& ctx = graphics::internal::context;

    // ---------- 1. СХЕМА НАБОРА (SetLayout) ----------
    // Описываем один слот: binding=0, тип uniform buffer, 1 штука,
    // доступен только в вершинном шейдере.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;                                  // совпадает с layout(binding=0) в шейдере
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings    = &binding;

    if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr,
                                    &vk_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout\n";
        return false;
    }

    // ---------- 2. ПУЛ ДЕСКРИПТОРОВ ----------
    // Пул — "фабрика". Здесь указываем, сколько дескрипторов какого типа
    // можно из него выделить. Нам нужен один uniform buffer.
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets       = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes    = &pool_size;

    if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr,
                               &vk_descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    // ---------- 3. ВЫДЕЛЕНИЕ КОНКРЕТНОГО НАБОРА ----------
    // Один набор по нашей схеме.
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool     = vk_descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts        = &vk_descriptor_set_layout;

    if (vkAllocateDescriptorSets(ctx.device, &set_info, &vk_descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set\n";
        return false;
    }

    // ---------- 4. ПРИВЯЗКА БУФЕРА К ДЕСКРИПТОРУ ----------
    // Указываем: конкретный uniform-буфер, с какого смещения и какого размера.
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = vk_uniform_buffer;
    buffer_info.offset = 0;
    buffer_info.range  = sizeof(GlobalUniforms);

    // Описываем операцию записи: в binding=0 набора vk_descriptor_set
    // положить buffer_info.
    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = vk_descriptor_set;
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo     = &buffer_info;

    // Выполняем запись.
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

    return true;
}

// Pipeline описывает весь путь
bool createPipeline() {
    auto& ctx = graphics::internal::context;

    // Загружаем скомпилированные шейдеры.
    VkShaderModule vert = loadShaderModule(SHADER_DIR "/parallelepiped.vert.spv");
    VkShaderModule frag = loadShaderModule(SHADER_DIR "/parallelepiped.frag.spv");
    if (!vert || !frag) {
        std::cerr << "Failed to load shaders\n";
        return false;
    }

    // ---------- СТАДИИ SHADER ----------
    // Вершинный и фрагментный — обязательны.
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";     // имя функции входа в GLSL

    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    // ---------- ОПИСАНИЕ ВЕРШИН ----------
    // Binding — откуда и как брать данные из вершинного буфера.
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(Vertex);                        // шаг между вершинами (24 байта)
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;           // новая вершина каждый шаг

    // Атрибуты — конкретные поля внутри вершины (позиция и цвет).
    std::array<VkVertexInputAttributeDescription, 2> attributes = {};
    // location=0 — позиция: 3 float, смещение 0 от начала вершины.
    attributes[0].location = 0;
    attributes[0].binding  = 0;
    attributes[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset   = offsetof(Vertex, position);
    // location=1 — цвет: 3 float, смещение 12 байт от начала вершины.
    attributes[1].location = 1;
    attributes[1].binding  = 0;
    attributes[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset   = offsetof(Vertex, color);

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount   = 1;
    vertex_input.pVertexBindingDescriptions      = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions    = attributes.data();

    // ---------- СБОРКА ПРИМИТИВОВ ----------
    // TRIANGLE_LIST — каждые 3 индекса = отдельный треугольник.
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // ---------- VIEWPORT / SCISSOR ----------
    // Говорим, что зададим их динамически в command buffer.
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount  = 1;

    // ---------- РАСТЕРИЗАЦИЯ ----------
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;                 // заливать треугольники
    raster.cullMode    = VK_CULL_MODE_NONE;                    // не отбрасывать грани (для отладки проще)
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;      // обход по часовой = лицевая
    raster.lineWidth   = 1.0f;

    // ---------- MULTISAMPLE ----------
    // Отключён (1 sample на пиксель).
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // ---------- DEPTH / STENCIL ----------
    // Включён depth test — иначе дальние грани рисовались бы поверх ближних.
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable  = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    // ---------- BLEND ----------
    // Просто пишем RGBA в framebuffer, без смешивания.
    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_attachment.blendEnable    = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments    = &color_attachment;

    // ---------- DYNAMIC STATE ----------
    // Какие состояния pipeline можно менять через команды (без пересоздания).
    VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates    = dynamic_states;

    // ---------- PIPELINE LAYOUT ----------
    // Связываем pipeline с описанием дескрипторных наборов. Без этого
    // шейдер не смог бы прочитать MVP из uniform-буфера.
    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts    = &vk_descriptor_set_layout;

    if (vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    // ---------- СОЗДАНИЕ PIPELINE ----------
    // Собираем все описания в одну структуру и создаём pipeline.
    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount          = 2;
    pipeline_info.pStages             = stages;
    pipeline_info.pVertexInputState   = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState      = &viewport_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState   = &multisample;
    pipeline_info.pDepthStencilState  = &depth_stencil;
    pipeline_info.pColorBlendState    = &color_blend;
    pipeline_info.pDynamicState       = &dynamic_state;
    pipeline_info.layout              = vk_pipeline_layout;
    pipeline_info.renderPass          = ctx.render_pass;    // pipeline совместим с этим render pass
    pipeline_info.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        vkDestroyShaderModule(ctx.device, vert, nullptr);
        vkDestroyShaderModule(ctx.device, frag, nullptr);
        return false;
    }

    // После создания pipeline шейдерные модули больше не нужны — можно уничтожить.
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);

    return true;
}

} // namespace (анонимное)

// Вызывается один раз при старте приложения (из main.cpp).
bool initialize() {
    if (!createBuffers())    return false;   // сначала буферы
    if (!createDescriptors()) return false;  // дескрипторы зависят от uniform-буфера
    if (!createPipeline())   return false;   // pipeline зависит от дескрипторов и шейдеров
    return true;
}

// Вызывается один раз при выходе. Уничтожаем всё в обратном порядке создания.
void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);     // ждём, пока GPU закончит работу

    // Уничтожаем Vulkan-объекты в порядке, обратном созданию.
    vkDestroyPipeline(ctx.device, vk_pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);
    vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_descriptor_set_layout, nullptr);

    // Уничтожаем буферы через VMA (она освобождает и буфер, и память).
    vmaDestroyBuffer(ctx.allocator, vk_uniform_buffer, vma_uniform_buffer_allocation);
    vmaDestroyBuffer(ctx.allocator, vk_index_buffer, vma_index_buffer_allocation);
    vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer, vma_vertex_buffer_allocation);
}

// Вызывается каждый кадр ДО рендера. Здесь: UI + построение MVP.
void update([[maybe_unused]] double time) {
    auto& ctx = graphics::internal::context;

    // ---------- IMGUI ПАНЕЛЬ ----------
    ImGui::Begin("Lab 1 - Controls");

    // Переключатель проекции.
    ImGui::Text("Projection");
    ImGui::RadioButton("Perspective",  &ui_use_perspective, 1); ImGui::SameLine();
    ImGui::RadioButton("Orthographic", &ui_use_perspective, 0);

    // Слайдер FOV активен только в перспективной проекции.
    if (ui_use_perspective == 1) {
        ImGui::SliderFloat("FOV", &ui_fov, 15.0f, 120.0f);
    }

    ImGui::Separator();

    // Слайдеры трансформаций. Каждый — три слайдера в одном (x, y, z).
    ImGui::Text("Transform");
    ImGui::SliderFloat3("Position", &ui_position.x, -5.0f, 5.0f);
    ImGui::SliderFloat3("Rotation", &ui_rotation.x, -180.0f, 180.0f);
    ImGui::SliderFloat3("Scale",    &ui_scale.x,    0.1f, 3.0f);

    // Кнопка сброса всех параметров в исходные.
    if (ImGui::Button("Reset")) {
        ui_position = { 0.0f, 0.0f, 0.0f };
        ui_rotation = { 0.0f, 0.0f, 0.0f };
        ui_scale    = { 1.0f, 1.0f, 1.0f };
        ui_use_perspective = 1;
        ui_fov = 45.0f;
    }

    ImGui::End();

    // ---------- ПОСТРОЕНИЕ MVP ----------
    // Aspect ratio нужно для корректной проекции.
    const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                         static_cast<float>(ctx.swapchain_extent.height);

    // Матрица Model: масштаб → повороты → смещение.
    // glm строит матрицу так, что последняя применённая функция
    // оказывается первой в цепочке преобразований вершины.
    // Поэтому "translate * rotate * scale" даёт: сначала scale, потом rotate, потом translate.
    glm::mat4 model = glm::mat4(1.0f);                          // единичная матрица
    model = glm::translate(model, ui_position);                 // смещение
    model = glm::rotate(model, glm::radians(ui_rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(ui_rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(ui_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, ui_scale);                        // масштаб

    // Матрица View: камера в точке (0,0,3), смотрит на (0,0,0), "вверх" — по Y.
    const glm::vec3 eye    = { 0.0f, 0.0f, 3.0f };
    const glm::vec3 target = { 0.0f, 0.0f, 0.0f };
    const glm::vec3 up     = { 0.0f, 1.0f, 0.0f };
    glm::mat4 view = glm::lookAt(eye, target, up);

    // Матрица Projection: выбирается пользователем через UI.
    glm::mat4 proj;
    if (ui_use_perspective == 1) {
        // Перспективная: дальние объекты меньше, есть деление на w.
        proj = glm::perspective(glm::radians(ui_fov), aspect, 0.1f, 100.0f);
    } else {
        // Ортографическая: размер не зависит от расстояния.
        const float half_size = 2.0f;
        proj = glm::ortho(-half_size * aspect, half_size * aspect,
                          -half_size, half_size,
                          0.1f, 100.0f);
    }

    // Итоговая матрица: применяется справа налево — Model, View, Projection.
    const glm::mat4 mvp = proj * view * model; // Формула MVP

    // Записываем MVP в uniform-буфер (память GPU, замапленная на CPU).
    if (uniform_buffer_mapped) {
        uniform_buffer_mapped->mvp = mvp;
    }
}

// Вызывается каждый кадр для записи команд отрисовки в command buffer.
void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    // Сбрасываем буфер и начинаем запись.
    vkResetCommandBuffer(fd.command_buffer, 0);
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(fd.command_buffer, &begin);

    // Значения для очистки: цвет фона (тёмно-серый) и depth=1.0 (максимум).
    const VkClearValue clear_values[2] = {
        { .color = {{ 0.1f, 0.1f, 0.1f, 1.0f }} },
        { .depthStencil = { 1.0f, 0 } },
    };

    // Начинаем render pass: связываем его с конкретным framebuffer'ом
    // и передаём clear-значения.
    const VkRenderPassBeginInfo render_pass_begin = {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass      = ctx.render_pass,
        .framebuffer     = fd.framebuffer,
        .renderArea      = { .extent = ctx.swapchain_extent },
        .clearValueCount = 2,
        .pClearValues    = clear_values,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

    // ---------- VIEWPORT ----------
    // Область экрана, в которую рисуем (весь экран).
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(ctx.swapchain_extent.width);
    viewport.height   = static_cast<float>(ctx.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

    // Scissor — прямоугольник, за пределами которого пиксели отсекаются.
    VkRect2D scissor{};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    // ---------- ПРИВЯЗКА РЕСУРСОВ ----------
    // 1. Pipeline — какой конвейер использовать.
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

    // 2. Вершинный буфер — откуда читать вершины.
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);

    // 3. Индексный буфер — как собирать треугольники.
    vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

    // 4. Дескрипторный набор — откуда шейдер возьмёт MVP-матрицу.
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk_pipeline_layout, 0, 1, &vk_descriptor_set, 0, nullptr);

    // ---------- КОМАНДА ОТРИСОВКИ ----------
    // Параметры: 36 индексов, 1 instance, смещение индексов 0,
    // смещение вершин 0, первый instance 0.
    vkCmdDrawIndexed(fd.command_buffer, 36, 1, 0, 0, 0);

    // Завершаем render pass и command buffer.
    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
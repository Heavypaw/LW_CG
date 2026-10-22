#version 450

// Входные данные из вершинного буфера
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Выходные данные для фрагментного шейдера
layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;

// Доп. 1: Set 0 = Scene (общие данные для всей сцены)
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 ambient_color;
    vec4 directional_direction;
    vec4 directional_color;
    vec4 counts;
} scene;

// Доп. 1: Set 1 = Model (специфично для текущего объекта)
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;
    vec4 specular_color;
} model_uniforms;

void main() {
    // Переводим вершину в мировое пространство
    vec4 world_pos = model_uniforms.model * vec4(in_position, 1.0);
    out_world_position = world_pos.xyz;

    // ВАЖНО: Нормаль нельзя просто умножить на матрицу модели, если объект 
    // имеет неравномерный масштаб (scale). Используем inverse-transpose.
    mat3 normal_matrix = mat3(transpose(inverse(model_uniforms.model)));
    out_world_normal = normal_matrix * in_normal;

    // Стандартное вычисление позиции на экране
    gl_Position = scene.view_projection * world_pos;
}
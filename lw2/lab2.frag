#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 0) out vec4 out_color;

// Set 0 (Scene)
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 ambient_color;
    vec4 directional_direction;
    vec4 directional_color;
    vec4 counts;
} scene;

// Set 1 (Model)
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;
    vec4 specular_color;
} model_uniforms;

// Доп. 2 и 3: Storage буферы для массивов источников (без указания размера!)
struct PointLight {
    vec4 position;
    vec4 color;      // rgb + intensity
    vec4 params;     // x = radius
};
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 params;     // x = radius, y = cos(inner), z = cos(outer)
};

layout(std430, set = 0, binding = 1) buffer PointLightsBuffer {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) buffer SpotLightsBuffer {
    SpotLight spot_lights[];
};

// ============================================================================
// ФУНКЦИЯ БЛИНН-ФОНГА 
// ============================================================================
vec3 blinn_phong(vec3 N, vec3 L, vec3 V, vec3 intensity,
                 vec3 albedo, vec3 spec_color, float shininess) {
    float NdotL = max(dot(N, L), 0.0); // Диффузная часть (основной цвет)
    vec3 H = normalize(L + V);         // Половинный вектор для блика
    float NdotH = max(dot(N, H), 0.0); // Угол для блика
    float spec = pow(NdotH, shininess); // Резкость блика
    return intensity * (albedo * NdotL + spec_color * spec);
}

void main() {
    vec3 N = normalize(in_world_normal);
    vec3 V = normalize(scene.camera_position.xyz - in_world_position); // Вектор на камеру
    vec3 albedo = model_uniforms.albedo.rgb;
    vec3 spec_color = model_uniforms.specular_color.rgb;
    float shininess = model_uniforms.specular_color.w;

    // 1. AMBIENT (Фоновый свет)
    vec3 result = scene.ambient_color.rgb * albedo;

    // 2. DIRECTIONAL LIGHT (Солнце)
    {
        vec3 L = normalize(-scene.directional_direction.xyz);
        vec3 intensity = scene.directional_color.rgb * scene.directional_color.w;
        result += blinn_phong(N, L, V, intensity, albedo, spec_color, shininess);
    }

    // 3. POINT LIGHTS (Доп. 2)
    int point_count = int(scene.counts.x);
    for (int i = 0; i < point_count; ++i) {
        vec3 to_light = point_lights[i].position.xyz - in_world_position;
        float d = length(to_light);
        vec3 L = to_light / max(d, 1e-4);
        
        vec3 light_color = point_lights[i].color.rgb;
        float light_int = point_lights[i].color.w;
        float radius = point_lights[i].params.x;

        // Затухание по закону обратных квадратов
        float atten = light_int / (d * d + 1.0);
        if (d > radius) atten = 0.0; // Отсечение за радиусом

        result += blinn_phong(N, L, V, light_color * atten, albedo, spec_color, shininess);
    }

    // 4. SPOT LIGHTS (Доп. 3)
    int spot_count = int(scene.counts.y);
    for (int i = 0; i < spot_count; ++i) {
        vec3 to_light = spot_lights[i].position.xyz - in_world_position;
        float d = length(to_light);
        vec3 L = to_light / max(d, 1e-4);

        vec3 spot_dir = normalize(spot_lights[i].direction.xyz);
        vec3 light_color = spot_lights[i].color.rgb;
        float light_int = spot_lights[i].color.w;
        float radius = spot_lights[i].params.x;
        
        // Берем косинусы углов (Inner и Outer)
        float cos_inner = max(spot_lights[i].params.y, spot_lights[i].params.z);
        float cos_outer = min(spot_lights[i].params.y, spot_lights[i].params.z);

        // Мягкие края: смотрим угол между направлением прожектора и лучом на пиксель
        float cos_angle = dot(-L, spot_dir);
        float denom = max(cos_inner - cos_outer, 1e-4);
        float spot_factor = clamp((cos_angle - cos_outer) / denom, 0.0, 1.0);

        // Затухание по расстоянию
        float atten = light_int / (d * d + 1.0);
        if (d > radius) atten = 0.0;

        result += blinn_phong(N, L, V, light_color * atten * spot_factor, albedo, spec_color, shininess);
    }

    out_color = vec4(result, 1.0);
}

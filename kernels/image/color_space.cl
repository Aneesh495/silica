#include "common/math.clh"
#include "common/memory.clh"

// Converts RGB image to Grayscale using ITU-R BT.709 luminance weights
__kernel void kernel_rgb_to_grayscale(
    __global const float* in_rgb,
    __global float* out_gray,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t in_idx = ((size_t)y * (size_t)width + (size_t)x) * 3;
    size_t out_idx = (size_t)y * (size_t)width + (size_t)x;

    float r = in_rgb[in_idx + 0];
    float g = in_rgb[in_idx + 1];
    float b = in_rgb[in_idx + 2];

    out_gray[out_idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// D65 Standard Illuminant reference white point
#define D65_X 0.95047f
#define D65_Y 1.00000f
#define D65_Z 1.08883f

inline float cielab_f(float t) {
    return (t > 0.008856f) ? cbrt(t) : (7.787f * t + 16.0f / 116.0f);
}

inline float cielab_finv(float t) {
    float t3 = t * t * t;
    return (t3 > 0.008856f) ? t3 : ((t - 16.0f / 116.0f) / 7.787f);
}

// Converts linear sRGB to CIELAB L*a*b*
__kernel void kernel_rgb_to_cielab(
    __global const float* in_rgb,
    __global float* out_lab,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 3;

    float r = in_rgb[idx + 0];
    float g = in_rgb[idx + 1];
    float b = in_rgb[idx + 2];

    // sRGB to XYZ matrix transform
    float X = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
    float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
    float Z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;

    float fx = cielab_f(X / D65_X);
    float fy = cielab_f(Y / D65_Y);
    float fz = cielab_f(Z / D65_Z);

    float L = 116.0f * fy - 16.0f;
    float a = 500.0f * (fx - fy);
    float b_val = 200.0f * (fy - fz);

    out_lab[idx + 0] = L;
    out_lab[idx + 1] = a;
    out_lab[idx + 2] = b_val;
}

// Converts RGB to HSV (Hue in [0, 360], Saturation in [0, 1], Value in [0, 1])
__kernel void kernel_rgb_to_hsv(
    __global const float* in_rgb,
    __global float* out_hsv,
    const int width,
    const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 3;

    float r = in_rgb[idx + 0];
    float g = in_rgb[idx + 1];
    float b = in_rgb[idx + 2];

    float max_val = fmax(fmax(r, g), b);
    float min_val = fmin(fmin(r, g), b);
    float delta = max_val - min_val;

    float h = 0.0f;
    float s = (max_val > EPSILON_F) ? (delta / max_val) : 0.0f;
    float v = max_val;

    if (delta > EPSILON_F) {
        if (fabs(max_val - r) < EPSILON_F) {
            h = 60.0f * fmod((g - b) / delta, 6.0f);
        } else if (fabs(max_val - g) < EPSILON_F) {
            h = 60.0f * (((b - r) / delta) + 2.0f);
        } else {
            h = 60.0f * (((r - g) / delta) + 4.0f);
        }
        if (h < 0.0f) h += 360.0f;
    }

    out_hsv[idx + 0] = h;
    out_hsv[idx + 1] = s;
    out_hsv[idx + 2] = v;
}

// Reinhard Global Tone Mapping with luminance preservation
__kernel void kernel_tone_map_reinhard(
    __global const float* in_hdr_rgb,
    __global float* out_ldr_rgb,
    const int width,
    const int height,
    const float white_point) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 3;

    float r = in_hdr_rgb[idx + 0];
    float g = in_hdr_rgb[idx + 1];
    float b = in_hdr_rgb[idx + 2];

    float L_in = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float L_out = (L_in * (1.0f + L_in / (white_point * white_point))) / (1.0f + L_in);

    float scale = (L_in > EPSILON_F) ? (L_out / L_in) : 0.0f;

    out_ldr_rgb[idx + 0] = clamp_f(r * scale, 0.0f, 1.0f);
    out_ldr_rgb[idx + 1] = clamp_f(g * scale, 0.0f, 1.0f);
    out_ldr_rgb[idx + 2] = clamp_f(b * scale, 0.0f, 1.0f);
}

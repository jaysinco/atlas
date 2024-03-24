#include "./fwd.cuh"
#include "./common.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "stb_image_write.h"

class vec3
{
public:
    __host__ __device__ vec3() {}

    __host__ __device__ vec3(float e0, float e1, float e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }

    inline __host__ __device__ float x() const { return e[0]; }

    inline __host__ __device__ float y() const { return e[1]; }

    inline __host__ __device__ float z() const { return e[2]; }

    inline __host__ __device__ float r() const { return e[0]; }

    inline __host__ __device__ float g() const { return e[1]; }

    inline __host__ __device__ float b() const { return e[2]; }

    inline __host__ __device__ const vec3& operator+() const { return *this; }

    inline __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    inline __host__ __device__ float operator[](int i) const { return e[i]; }

    inline __host__ __device__ float& operator[](int i) { return e[i]; };

    inline __host__ __device__ vec3& operator+=(vec3 const& v2);
    inline __host__ __device__ vec3& operator-=(vec3 const& v2);
    inline __host__ __device__ vec3& operator*=(vec3 const& v2);
    inline __host__ __device__ vec3& operator/=(vec3 const& v2);
    inline __host__ __device__ vec3& operator*=(float const t);
    inline __host__ __device__ vec3& operator/=(float const t);

    inline __host__ __device__ float length() const
    {
        return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }

    inline __host__ __device__ float squared_length() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    inline __host__ __device__ void make_unit_vector();

    float e[3];
};

inline std::istream& operator>>(std::istream& is, vec3& t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, vec3 const& t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

inline __host__ __device__ void vec3::make_unit_vector()
{
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

inline __host__ __device__ vec3 operator+(vec3 const& v1, vec3 const& v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline __host__ __device__ vec3 operator-(vec3 const& v1, vec3 const& v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline __host__ __device__ vec3 operator*(vec3 const& v1, vec3 const& v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

inline __host__ __device__ vec3 operator/(vec3 const& v1, vec3 const& v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline __host__ __device__ vec3 operator*(float t, vec3 const& v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline __host__ __device__ vec3 operator/(vec3 v, float t)
{
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

inline __host__ __device__ vec3 operator*(vec3 const& v, float t)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline __host__ __device__ float dot(vec3 const& v1, vec3 const& v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

inline __host__ __device__ vec3 cross(vec3 const& v1, vec3 const& v2)
{
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]), (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

inline __host__ __device__ vec3& vec3::operator+=(vec3 const& v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

inline __host__ __device__ vec3& vec3::operator*=(vec3 const& v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

inline __host__ __device__ vec3& vec3::operator/=(vec3 const& v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

inline __host__ __device__ vec3& vec3::operator-=(vec3 const& v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

inline __host__ __device__ vec3& vec3::operator*=(float const t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

inline __host__ __device__ vec3& vec3::operator/=(float const t)
{
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

inline __host__ __device__ vec3 unit_vector(vec3 v) { return v / v.length(); }

class ray
{
public:
    __device__ ray() {}

    __device__ ray(vec3 const& a, vec3 const& b)
    {
        A = a;
        B = b;
    }

    __device__ vec3 origin() const { return A; }

    __device__ vec3 direction() const { return B; }

    __device__ vec3 point_at_parameter(float t) const { return A + t * B; }

    vec3 A;
    vec3 B;
};

__device__ bool hit_sphere(vec3 const& center, float radius, ray const& r)
{
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ vec3 color(ray const& r)
{
    if (hit_sphere(vec3(0, 0, -1), 0.5, r)) return vec3(1, 0, 0);
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(uint8_t* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal,
                       vec3 vertical, vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    vec3 c = color(r);
    int offset = j * max_x * 3 + i * 3;
    fb[offset + 0] = 255.99 * c.r();
    fb[offset + 1] = 255.99 * c.g();
    fb[offset + 2] = 255.99 * c.b();
}

int ray_tracing(int argc, char** argv)
{
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * 3;

    // allocate FB
    uint8_t* fb;
    CHECK(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, nx, ny, vec3(-2.0, -1.0, -1.0), vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 0.0));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    stbi_write_jpg("ray_tracing.jpg", nx, ny, 3, fb, 100);
    CHECK(cudaFree(fb));

    return 0;
}

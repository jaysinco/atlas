#include "./common.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "toolkit/toolkit.h"

class Vec3
{
public:
    __host__ __device__ Vec3() = default;

    __host__ __device__ Vec3(float e0, float e1, float e2)
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

    inline __host__ __device__ Vec3 const& operator+() const { return *this; }

    inline __host__ __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }

    inline __host__ __device__ float operator[](int i) const { return e[i]; }

    inline __host__ __device__ float& operator[](int i) { return e[i]; };

    inline __host__ __device__ Vec3& operator+=(Vec3 const& v);
    inline __host__ __device__ Vec3& operator-=(Vec3 const& v);
    inline __host__ __device__ Vec3& operator*=(Vec3 const& v);
    inline __host__ __device__ Vec3& operator/=(Vec3 const& v);
    inline __host__ __device__ Vec3& operator*=(float const t);
    inline __host__ __device__ Vec3& operator/=(float const t);

    inline __host__ __device__ float length() const
    {
        return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }

    inline __host__ __device__ float squaredLength() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    inline __host__ __device__ void makeUnitVector();

    float e[3];
};

inline std::istream& operator>>(std::istream& is, Vec3& t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, Vec3 const& t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

inline __host__ __device__ void Vec3::makeUnitVector()
{
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

inline __host__ __device__ Vec3 operator+(Vec3 const& v1, Vec3 const& v2)
{
    return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline __host__ __device__ Vec3 operator-(Vec3 const& v1, Vec3 const& v2)
{
    return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline __host__ __device__ Vec3 operator*(Vec3 const& v1, Vec3 const& v2)
{
    return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

inline __host__ __device__ Vec3 operator/(Vec3 const& v1, Vec3 const& v2)
{
    return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline __host__ __device__ Vec3 operator*(float t, Vec3 const& v)
{
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline __host__ __device__ Vec3 operator/(Vec3 v, float t)
{
    return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

inline __host__ __device__ Vec3 operator*(Vec3 const& v, float t)
{
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline __host__ __device__ float dot(Vec3 const& v1, Vec3 const& v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

inline __host__ __device__ Vec3 cross(Vec3 const& v1, Vec3 const& v2)
{
    return Vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]), (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

inline __host__ __device__ Vec3& Vec3::operator+=(Vec3 const& v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

inline __host__ __device__ Vec3& Vec3::operator*=(Vec3 const& v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

inline __host__ __device__ Vec3& Vec3::operator/=(Vec3 const& v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

inline __host__ __device__ Vec3& Vec3::operator-=(Vec3 const& v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

inline __host__ __device__ Vec3& Vec3::operator*=(float const t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

inline __host__ __device__ Vec3& Vec3::operator/=(float const t)
{
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

inline __host__ __device__ Vec3 unitVector(Vec3 v) { return v / v.length(); }

class Ray
{
public:
    __device__ Ray() = default;

    __device__ Ray(Vec3 const& a, Vec3 const& b)
    {
        this->a = a;
        this->b = b;
    }

    __device__ Vec3 origin() const { return a; }

    __device__ Vec3 direction() const { return b; }

    __device__ Vec3 pointAtParameter(float t) const { return a + t * b; }

    Vec3 a;
    Vec3 b;
};

__device__ bool hitSphere(Vec3 const& center, float radius, Ray const& r)
{
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ Vec3 color(Ray const& r)
{
    if (hitSphere(Vec3(0, 0, -1), 0.5, r)) {
        return Vec3(1, 0, 0);
    }
    Vec3 unit_direction = unitVector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
}

__global__ void render(uint8_t* fb, int max_x, int max_y, Vec3 lower_left_corner, Vec3 horizontal,
                       Vec3 vertical, Vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) {
        return;
    }
    float u = static_cast<float>(i) / static_cast<float>(max_x);
    float v = static_cast<float>(j) / static_cast<float>(max_y);
    Ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    Vec3 c = color(r);
    int offset = j * max_x * 3 + i * 3;
    fb[offset + 2] = 255.99 * c.r();
    fb[offset + 1] = 255.99 * c.g();
    fb[offset + 0] = 255.99 * c.b();
}

int rayTracing(int argc, char** argv)
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
    render<<<blocks, threads>>>(fb, nx, ny, Vec3(-2.0, -1.0, -1.0), Vec3(4.0, 0.0, 0.0),
                                Vec3(0.0, 2.0, 0.0), Vec3(0.0, 0.0, 0.0));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = (static_cast<double>(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    cv::Mat img(ny, nx, CV_8UC3, fb);
    cv::imwrite((toolkit::getTempDir() / "ray_tracing.jpg").string(), img);
    CHECK(cudaFree(fb));

    return 0;
}

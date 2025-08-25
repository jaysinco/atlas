#version 450

layout(set = 0, binding = 0) buffer InBuffer {
    int data[];
} inBuffer;

layout(set = 0, binding = 1) buffer OutBuffer {
    int data[];
} outBuffer;

layout(push_constant) uniform PushConstants
{
    ivec4 data;
} constants;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint index = gl_GlobalInvocationID.x;
    outBuffer.data[index] = inBuffer.data[index] * 2 + constants.data.y;
}

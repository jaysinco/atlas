#version 450
// #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// #extension GL_EXT_buffer_reference2 : require

layout(set = 0, binding = 0) buffer InBuffer {
    int data[];
} inBuffer;

layout(set = 0, binding = 1) buffer OutBuffer {
    int data[];
} outBuffer;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint index = gl_GlobalInvocationID.x;
    outBuffer.data[index] = inBuffer.data[index] * inBuffer.data[index];
}

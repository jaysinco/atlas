#pragma once
#include <cstdint>

int hello_world(int argc, char** argv);
int check_device(int argc, char** argv);
int sum_matrix(int argc, char** argv);
int reduce_integer(int argc, char** argv);
int nested_hello_world(int argc, char** argv);
int global_variable(int argc, char** argv);
int test_cufft(int argc, char** argv);
int julia_set(int argc, char** argv);
void fill_julia_set(int image_width, int image_height, uint8_t* pixels);
int dot_product(int argc, char** argv);
int ray_tracing(int argc, char** argv);
int txi_gaussian(int argc, char** argv);

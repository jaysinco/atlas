# Using Nsight Systems to profile GPU workload

* Nsight Compute
```
sudo ncu -o profile <exe-path>
ncu-ui ./profile.ncu-rep
```

* Nsight Systems
```
sudo nsys profile --trace=cuda <exe-path>
nsys-ui ./report1.nsys-rep
```

# GPU Benchmark
* txi gaussian param
```
image_size = 1920x1080x4
radius = 8
win_size = radius * 2 + 1
sigma = 100
enhance_k = 1.5
output_mode = 'texture'
```
* maximized jetson performance
```
sudo nvpmodel -m 0
sudo jetson_clocks
```
* result table

| SOC | GPU | OS | ARCH | API | Image Size (bytes) | Kernel Run (ms) | Host To Device Copy (ms) | Device To Host Copy (ms) |
| --- | --- | -- | ---- | --- | ------------------ | --------------- | ------------------------ | ------------------------ |
| RK3588 | Mali-G610 MP4 | Linux | aarch64 | OpenCL 2.1 | 8294400 | 34.2 | 3.3 | 2.7 |
| Jetson AGX Orin 64GB | Ampere GPU | Linux | aarch64 | CUDA 11.4 | 8294400 | 2.8 | 0.8 | 1.9 |
| / | RTX 3050 Laptop | Windows | x86_64 | CUDA 11.4 | 8294400 | 1.9 | 1.4 | 1.7 |
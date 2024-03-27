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
* result table

| SOC | GPU | OS | ARCH | API | Image Size (bytes) | Kernel Run (ms) | Host To Device Copy (ms) | Device To Host Copy (ms) |
| --- | --- | -- | ---- | --- | ------------------ | --------------- | ------------------------ | ------------------------ |
| RK3588 | Mali-G610 MP4 | Linux | aarch64 | OpenCL | 8294400 | 34.2 | 3.3 | 2.7 |
| Jetson AGX Orin 64GB | Ampere GPU | Linux | aarch64 | CUDA | 8294400 | 4.4 | 1.3 | 2.2 |
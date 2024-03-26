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

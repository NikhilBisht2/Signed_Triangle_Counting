# Compilation
```bash
g++ -O3 -march=native -fopenmp <filename> -o <binary_name>
```
## Execution
Using all available CPU cores:
```bash
OMP_NUM_THREADS=$(nproc) ./balanced_split <graph_file>
```
## Node HPC
```bash
srun -p gpu_rtx_pro_6000_6_csis_hyd --gres=gpu:1 --pty bash
module purge
module load gcc-11.2.0-gcc-8.5.0-ov3qrz6
module load cuda-12.1.0-gcc-11.2.0-s5o57xp
```
grpah links:
```bash
https://sparse.tamu.edu/MAWI/mawi_201512012345
https://sparse.tamu.edu/GAP/GAP-road
https://sparse.tamu.edu/DIMACS10/oh2010
https://sparse.tamu.edu/DIMACS10/tx2010
```
BUILD OFF:
```bash
cmake .. -DBUILD_TESTING=OFF
make -j$(nproc)
```

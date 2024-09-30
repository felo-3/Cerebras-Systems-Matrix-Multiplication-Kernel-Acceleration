1. To execute the implementation of cannon's algorithm on the CPU, first log into the USC discovery server, then compile the code using the following command:

mpicc -o cannon cannon_updated.c

Then run the following command to execute the program:

sbatch cannon.sl

To adjust parameters of the program, go to cannon_updated.c:
To adjust the number of processors, modify int p on line 12. Note that this specifies one dimension of the 2D processors mesh. - then go to cannon.sl: - adjust the number before "cannon" to the total number of processors.
e.g. a 8 x 8 mesh
p = 8 //cannon_updated.c
mpirun --oversubscribe -np 64 cannon //cannon.sl
To adjust the matrix size, modify int m on line 11. - e.g. matrix size = 32 x 32
m = 1024

2. To execute the Matrix Multiplication algorithm implemented on the GPU, first log into the USC discovery server. Then compile the code using the following commands:
   module load nvidia-hpc-sdk
   nvcc p2 add -o p2.cu

then run the following command to execute the program

sbatch job_p2.sl

To adjust the thread block, block size and matrix size, adjust the defined variables P, block_size and size respectively.

3. To execute the implementation of SUMMA on Cerebras, you must have Cerebras SDK 1.1.0 simulator installed on your device:

Go to the directory where your files are and run the following command:
./commands.sh

The matrix dimension (NxN) is a product of the P value and the Mt,Kt,Nt value. For consistency, Mt, Kt, Nt should always be equal to represent square matrices. N = P\*Kt
When changing the P value, the fabric-dims should also be changed to accommodate the new PE mesh. For any P value, make the fabric-dims=p+7,p+2

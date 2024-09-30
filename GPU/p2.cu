#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define size 224
#define block_size 14
#define P 16

__global__ void matrix_mult(int *a, int *b, int *c)
{
    int row = threadIdx.x;
    int col = threadIdx.y;

    int my_x = blockIdx.x * blockDim.x + threadIdx.x;
    int my_y = blockIdx.y * blockDim.y + threadIdx.y;

    int i, j;
    int local_c = 0;

    __shared__ int a_shared[P][P];
    __shared__ int b_shared[P][P];

    for (i = 0; i < size / block_size; i++)
    {
        a_shared[row][col] = a[my_x * size + (i * block_size + col)];
        b_shared[row][col] = b[(i * block_size + row) * size + my_y];
        __syncthreads();
        for (j = 0; j < block_size; j++)
        {
            local_c += a_shared[row][j] * b_shared[j][col];
        }
        __syncthreads();
    }

    c[my_x * size + my_y] = local_c;
}

int main()
{
    int i;
    int *a = (int *)malloc(sizeof(int) * size * size);
    int *b = (int *)malloc(sizeof(int) * size * size);
    int *c = (int *)malloc(sizeof(int) * size * size);

    for (i = 0; i < size * size; i++)
    {
        a[i] = 1;
        b[i] = 2;
        c[i] = 0;
    }

    int *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void **)&gpu_a, sizeof(int) * size * size);
    cudaMalloc((void **)&gpu_b, sizeof(int) * size * size);
    cudaMalloc((void **)&gpu_c, sizeof(int) * size * size);

    struct timespec start, stop;
    double time;

    cudaMemcpy(gpu_a, a, sizeof(int) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(int) * size * size, cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1);  // 2 dimensional grid
    dim3 dimBlock(P, P); // 2 dimensional blocks

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }
    matrix_mult<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
    cudaMemcpy(c, gpu_c, sizeof(int) * size * size, cudaMemcpyDeviceToHost);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("time is %f ns\n", time * 1e9);

    printf("c[%d][%d]=%d ", 451, 451, c[451 * size + 451]);

    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    return 0;
}
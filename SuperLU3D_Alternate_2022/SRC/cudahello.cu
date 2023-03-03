#include <stdio.h>
__global__ void hello()
{
    printf("Hello world from device");
}

// int main()
// {
//     hello<<<32,32>>>();
//     cudadevicesynchronize();
//     return 0;
// }

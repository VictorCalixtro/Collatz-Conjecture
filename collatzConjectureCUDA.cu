#include <cstdio>
#include <algorithm>
#include <cuda.h>
#include <sys/time.h>

static const int ThreadsPerBlock = 512;

static __global__ void collatz(const long upper, int* const maxlen)
{
  const long i = threadIdx.x + blockIdx.x * (long)blockDim.x;
  // compute sequence lengths
  if (i < (upper + 1)/2) {
    long val = 2*i + 1; // translate to i-th odd
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val = val / 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
    if(len > *maxlen) atomicMax(maxlen, len); // per instructions
  }
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.2\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);}
  const long upper = atol(argv[1]);
  if (upper < 5) {fprintf(stderr, "ERROR: upper_bound must be at least 5\n"); exit(-1);}
  if ((upper % 2) != 1) {fprintf(stderr, "ERROR: upper_bound must be an odd number\n"); exit(-1);}
  printf("upper bound: %ld\n", upper);

  // allocate cpu vars
  int* const maxlen = new int;
  *maxlen = 0;

  // allocate gpu vars
  int* d_maxlen;
  if (cudaSuccess != cudaMalloc((void **)&d_maxlen, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  // initialize gpu vars
  if (cudaSuccess != cudaMemcpy(d_maxlen, maxlen, sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  // because we're only testing odd values, there are (upper+1)/2 number of iterations
  collatz<<<((upper+1)/2 + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(upper, d_maxlen);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // get result from GPU
  CheckCuda();
  if (cudaSuccess != cudaMemcpy(maxlen, d_maxlen, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying to host failed\n"); exit(-1);}

  // print result
  printf("longest sequence: %d elements\n", *maxlen);

  // clean up
  free(maxlen);
  cudaFree(d_maxlen);

  return 0;
}

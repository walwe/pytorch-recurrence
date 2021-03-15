#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
namespace {
    /**
    * CUDA Kernel to calculate Recurrence Matrix for full batch
    */
    template <typename scalar_t>
    __global__ void recurrence_kernel(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> in,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> out,
        const int batchSize,
        const int size) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int batch = blockIdx.z;

        if(i<size && j<size) {
          out[batch][0][j][i] = abs(in[batch][i]-in[batch][j]);
        }
    }
} // namespace

/**
* Calculate Recurrence Matrix for full batch
*/
void recurrence_matrix_calc(torch::Tensor in, torch::Tensor output, int batchSize, int size) {
    const int threads = 8;
    const dim3 threadsPerBlock(threads, threads);
    const int blockSize = ceil(size/float(threads));
    const dim3 numBlocks(blockSize, blockSize, batchSize);

    // Run kernel using tensor's data type
    AT_DISPATCH_ALL_TYPES(in.type(), "recurrence_kernel", ([&] {
       recurrence_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            in.packed_accessor<scalar_t,2,torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits, size_t>(),
            batchSize, size
       );
    }));

   auto err = cudaGetLastError();
   if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
   }
}
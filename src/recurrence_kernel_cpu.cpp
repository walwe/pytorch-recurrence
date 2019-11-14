// NOTE: This is a copy of cuda_extension_kernel.cu. It's kept here to test
// collision handling when a C++ file and CUDA file share the same filename.
// Setuptools can't deal with this at all, so the setup.py-based test uses
// cuda_extension_kernel.cu and the JIT test uses this file. Symlinks don't
// work well on Windows, so this is the most thorough solution right now.

#include <torch/extension.h>

#include <ATen/ATen.h>
namespace {
    template <typename scalar_t>
    void recurrence_kernel(
        const torch::TensorAccessor<scalar_t,2> in,
        torch::TensorAccessor<scalar_t,4> out,
        const int batchSize,
        const int size) {
        //
        // Slow single thread implementation
        // CPU version was just implemented for debugging purposes
        //
        for(int b=0; b<batchSize; b++){
            for(int i=0; i<size; i++) {
                for(int j=0; j<size; j++) {
                    out[b][0][j][i] = abs(in[b][i]-in[b][j]);
                }
            }
        }
    }
} // namespace

void recurrence_matrix_calc(torch::Tensor in, torch::Tensor output, int batchSize, int size) {

    // Run kernel using tensor's data type
    AT_DISPATCH_ALL_TYPES(in.type(), "recurrence_kernel", ([&] {
       recurrence_kernel<scalar_t>(
            in.accessor<scalar_t,2>(),
            output.accessor<scalar_t,4>(),
            batchSize, size
       );
    }));

}
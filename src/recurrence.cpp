#include <torch/extension.h>

void recurrence_matrix_calc(torch::Tensor in, torch::Tensor output, int batchSize, int size);

torch::Tensor recurrence_matrix(torch::Tensor x) {

    int expectedDim = 2;
    if(x.dim() != expectedDim) {
        fprintf(stderr, "Tensor must have dim %i, and has %li instead\n", expectedDim, x.dim());
        exit(-1);
    }

    int batchSize = x.size(0);
    int inputSize = x.size(1);
    int channels = 1;

    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());

    auto output = torch::empty({batchSize, channels, inputSize, inputSize}, options);

    recurrence_matrix_calc(x, output, batchSize, inputSize);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("recurrence_matrix", &recurrence_matrix, "recurrence_matrix(x)");
}
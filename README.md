# PyTorch Recurrence Matrix / Plot Operator
PyTorch operator to calculate unthresholded [Recurrence Matrix / Plot](https://en.wikipedia.org/wiki/Recurrence_plot) from Tensors on CPU/GPU.

> A recurrence plot (RP) is an advanced technique of nonlinear data analysis. It is a visualisation (or a graph) of a square matrix, in which the matrix elements correspond to those times at which a state of a dynamical system recurs (columns and rows correspond then to a certain pair of times). Techniqually, the RP reveals all the times when the phase space trajectory of the dynamical system visits roughly the same area in the phase space.
> 
> Source: [recurrence-plot.tk](http://www.recurrence-plot.tk/glance.php)
> 

Although a CPU implementation is available, this is just for debugging, because it is non optimized. 
If CUDA is not present, the CPU version will be compiled.

## Requirements
- CUDA
- pytorch
- CUDA device (GPU)

## Build & Install
The pytorch operator can be directly build and installed using the system python or within a docker container

### Build and install using system python
```
python setup.py install
```

### Build wheel in docker container

```
docker build --tag pytorch_recurrence_build .
docker run --rm -v /tmp/out:/out pytorch_recurrence_build
```
Wheel file can be found in `/tmp/out`

Install wheel
```
pip install /tmp/out/*.whl
```
## Usage

Define Recurrence layer
```
import torch
from recurrence_matrix import recurrence_matrix


class RecurrenceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return recurrence_matrix(x)


class RecurrenceLayer(torch.nn.Module):

    @staticmethod
    def forward(x):
        return RecurrenceFunction.apply(x)
```
Add layer to existing image consuming model

```
model = torch.nn.Sequential(RecurrenceLayer(), image_model)
```

Resize image using MaxPooling
```
image_size = 224
resize_factor = int(num_inputs/image_size)
if resize_factor < 1:
    resize_factor = 1

model = torch.nn.Sequential(
    RecurrenceLayer(),
    torch.nn.Conv2d(1, network_channels, padding=3, kernel_size=7, bias=False),
    torch.nn.MaxPool2d(resize_factor),
    image_model
)
```

## License
MIT licensed as found in the [LICENSE](LICENSE) file.

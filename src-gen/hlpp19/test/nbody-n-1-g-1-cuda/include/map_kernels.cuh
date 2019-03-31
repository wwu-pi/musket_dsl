

#pragma once


namespace mkt {

namespace kernel {

// DECLARATION

template <typename T, typename R, typename F>
__global__ void mapKernel(T* in,
                          R* out,
                          size_t size,
                          F func);

template <typename T, typename F>
__global__ void mapInPlaceKernel(T* inout,
                        size_t size,
                        F func);


template <typename T, typename R, typename F>
__global__ void mapIndexKernel(T* in,
                               R* out,
                               size_t size,
                               size_t offset,
                               F func);

template <typename T, typename F>
__global__ void mapIndexInPlaceKernel(T* inout,
                               size_t size,
                               size_t offset,
                               F func);

} // namespace kernel
} // namespace mkt

// DEFINITION

template <typename T, typename R, typename F>
__global__ void mkt::kernel::mapKernel(T* in,
                                       R* out,
                                       size_t size,
                                       F func)
{
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < size) {
    out[x] = func(in[x]);
  }
}

template <typename T, typename F>
__global__ void mkt::kernel::mapInPlaceKernel(T* inout,
                                       size_t size,
                                       F func)
{
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < size) {
    func(inout[x]);
  }
}

template <typename T, typename R, typename F>
__global__ void mkt::kernel::mapIndexKernel(T* in,
                                            R* out,
                                            size_t size,
                                            size_t offset,
                                            F func)
{
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < size) {
    out[x] = func(x + offset, in[x]);
  }
}

template <typename T, typename F>
__global__ void mkt::kernel::mapIndexInPlaceKernel(T* inout,
                                            size_t size,
                                            size_t offset,
                                            F func)
{
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < size) {
    func(x + offset, inout[x]);
  }
}


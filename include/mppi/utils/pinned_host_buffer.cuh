#pragma once

#include <mppi/utils/gpu_err_chk.cuh>

#include <cstddef>

namespace mppi
{
namespace memory
{

/**
 * @brief A fixed-capacity host buffer backed by CUDA page-locked memory.
 *
 * The buffer is intentionally non-copyable: its address must remain stable while asynchronous
 * transfers are queued against it. Resizing is expected to happen only when the owning CUDA
 * object's problem dimensions change, never in its steady-state execution path.
 */
template <class T>
class PinnedHostBuffer
{
public:
  PinnedHostBuffer() = default;

  ~PinnedHostBuffer()
  {
    reset();
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer(PinnedHostBuffer&&) = delete;
  PinnedHostBuffer& operator=(PinnedHostBuffer&&) = delete;

  void resize(const std::size_t size)
  {
    if (size == size_)
    {
      return;
    }

    reset();
    if (size > 0)
    {
      void* allocation = nullptr;
      HANDLE_ERROR(cudaMallocHost(&allocation, size * sizeof(T)));
      data_ = static_cast<T*>(allocation);
      size_ = size;
    }
  }

  void reset()
  {
    if (data_ != nullptr)
    {
      HANDLE_ERROR(cudaFreeHost(data_));
      data_ = nullptr;
      size_ = 0;
    }
  }

  T* data()
  {
    return data_;
  }

  const T* data() const
  {
    return data_;
  }

  std::size_t size() const
  {
    return size_;
  }

private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace memory
}  // namespace mppi

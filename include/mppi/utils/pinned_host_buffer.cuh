#pragma once

#include <mppi/utils/gpu_err_chk.cuh>

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

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
  static_assert(std::is_nothrow_destructible<T>::value, "Pinned buffer elements must have noexcept destructors");

public:
  PinnedHostBuffer() = default;

  ~PinnedHostBuffer() noexcept
  {
    release(false);
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
      try
      {
        for (; size_ < size; ++size_)
        {
          ::new (static_cast<void*>(data_ + size_)) T{};
        }
      }
      catch (...)
      {
        release(false);
        throw;
      }
    }
  }

  void reset()
  {
    release(true);
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
  void release(const bool throw_on_error)
  {
    T* allocation = std::exchange(data_, nullptr);
    std::size_t constructed = std::exchange(size_, 0);
    if (allocation != nullptr)
    {
      while (constructed > 0)
      {
        allocation[--constructed].~T();
      }
      gpuAssert(cudaFreeHost(allocation), __FILE__, __LINE__, throw_on_error);
    }
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace memory
}  // namespace mppi

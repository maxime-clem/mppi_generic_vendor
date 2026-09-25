#pragma once

#include <cuda_runtime.h>

namespace mppi
{
namespace safety
{
// Keep legacy status 0/1 meaningful. Low bits accumulate reasons; upper bits retain
// the earliest event, ordered by timestep, reason, then geometry index. This fits
// in the existing per-rollout int and supports both serial and split-cost kernels.
constexpr int kGeneric = 1;
constexpr int kLateral = 2;
constexpr int kObstacle = 4;
constexpr int kRoadBorder = 8;
constexpr int kReasonMask = 15;

__host__ __device__ inline int merge(const int a, const int b)
{
  const int event_a = a & ~kReasonMask;
  const int event_b = b & ~kReasonMask;
  const int event = event_a == 0 ? event_b : (event_b == 0 || event_a < event_b ? event_a : event_b);
  return event | ((a | b) & kReasonMask);
}

// reason: 1=lateral, 2=obstacle, 3=road border. index=-1 means not applicable.
// Unsupported metadata retains the safety flag, without fabricating an identity/time.
__host__ __device__ inline int event(const int reason, const int timestep, const int index = -1)
{
  if (reason < 1 || reason > 3)
    return kGeneric;
  const int flag = 1 << reason;
  if (timestep < 0 || timestep > 1022 || index < -1 || index > 1022)
    return flag;
  return ((((timestep + 1) << 12) | (reason << 10) | (index + 1)) << 4) | flag;
}

__host__ __device__ inline int timestep(const int status)
{
  return (status >> 16) - 1;
}
__host__ __device__ inline int reason(const int status)
{
  return (status >> 14) & 3;
}
__host__ __device__ inline int geometryIndex(const int status)
{
  return ((status >> 4) & 1023) - 1;
}
}  // namespace safety
}  // namespace mppi

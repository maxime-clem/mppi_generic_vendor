#pragma once

#include <mppi/core/mppi_common.cuh>

#include <algorithm>
#include <cmath>

namespace mppi
{
namespace controllers
{
/** Adapt informative, eligible weights; bound each cycle's temperature change to [1/2, 2]. */
inline float adaptEssLambda(float lambda, const kernels::CostWeightStats& stats, float target_ratio, float gain,
                            float lambda_min, float lambda_max)
{
  const float current = std::max(lambda_min, std::min(lambda_max, lambda));
  const double target = static_cast<double>(target_ratio) * stats.eligible_count;
  if (gain == 0.0F || stats.eligible_count < 2 || !std::isfinite(stats.normalizer) || stats.normalizer <= 0.0F ||
      !std::isfinite(stats.effective_sample_size) || stats.effective_sample_size <= 0.0F ||
      static_cast<double>(stats.normalization_upper_cost) - stats.rollout_min_cost < 1.0E-6 ||
      stats.minimum_cost_count >= stats.eligible_count || target <= stats.minimum_cost_count)
  {
    // Tied minima impose an ESS floor. Cooling cannot reach an infeasible target, and equal costs
    // give no temperature information. Do not wind up against lambda_min in either case.
    return current;
  }
  const double log_limit = std::log(2.0);
  const double log_step = std::max(
      -log_limit, std::min(log_limit, static_cast<double>(gain) * std::log(target / stats.effective_sample_size)));
  const double next = static_cast<double>(current) * std::exp(log_step);
  return static_cast<float>(std::max(static_cast<double>(lambda_min), std::min(static_cast<double>(lambda_max), next)));
}
}  // namespace controllers
}  // namespace mppi

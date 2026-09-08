#include <gtest/gtest.h>
#include <mppi/controllers/MPPI/ess_lambda_adaptation.h>
#include <kernel_tests/core/normexp_kernel_test.cuh>
#include <mppi/utils/test_helper.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

class NormExpKernel : public testing::Test
{
protected:
  void SetUp() override
  {
    generator = std::default_random_engine(7.0);
    distribution = std::normal_distribution<float>(100.0, 2.0);
  }

  void TearDown() override
  {
  }

  std::default_random_engine generator;
  std::normal_distribution<float> distribution;
};

template <int NUM_ROLLOUTS>
__global__ void computeNormalizerKernel(const float* __restrict__ costs, float* __restrict__ output)
{
  __shared__ float reduction_buffer[NUM_ROLLOUTS];
  int global_idx = threadIdx.x;
  int global_step = blockDim.x;
  *output = mppi::kernels::computeNormalizer(NUM_ROLLOUTS, costs, reduction_buffer, global_idx, global_step);
};

template <int NUM_ROLLOUTS>
__global__ void computeBaselineCostKernel(const float* __restrict__ costs, float* __restrict__ output)
{
  __shared__ float reduction_buffer[NUM_ROLLOUTS];
  int global_idx = threadIdx.x;
  int global_step = blockDim.x;
  *output = mppi::kernels::computeBaselineCost(NUM_ROLLOUTS, costs, reduction_buffer, global_idx, global_step);
};

TEST_F(NormExpKernel, computeBaselineCost_Test)
{
  const int num_rollouts = 4196;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float min_cost_known = *std::min_element(cost_vec.begin(), cost_vec.end());
  float min_cost_compute = mppi::kernels::computeBaselineCost(cost_vec.data(), num_rollouts);

  ASSERT_FLOAT_EQ(min_cost_compute, min_cost_known);
}

TEST_F(NormExpKernel, computeNormalizer_Test)
{
  const int num_rollouts = 1024;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float sum_cost_known = std::accumulate(cost_vec.begin(), cost_vec.end(), 0.0);
  float sum_cost_compute = mppi::kernels::computeNormalizer(cost_vec.data(), num_rollouts);

  ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

TEST_F(NormExpKernel, computeNormalizerDevice_Test)
{
  const int num_rollouts = 6048;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (int i = 0; i < cost_vec.size(); i++)
  {
    cost_vec[i] = distribution(generator);
  }
  float* norm_d;
  float* costs_d;
  float sum_cost_compute;
  HANDLE_ERROR(cudaMalloc((void**)&norm_d, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMemcpy(costs_d, cost_vec.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));
  computeNormalizerKernel<num_rollouts><<<1, 1024>>>(costs_d, norm_d);
  HANDLE_ERROR(cudaMemcpy(&sum_cost_compute, norm_d, sizeof(float), cudaMemcpyDeviceToHost));

  float sum_cost_known = std::accumulate(cost_vec.begin(), cost_vec.end(), 0.0);
  ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

TEST_F(NormExpKernel, computeBaselineCostDevice_Test)
{
  const int num_rollouts = 6048;
  std::array<float, num_rollouts> cost_vec = { 0 };

  // Use a range based for loop to set the cost
  for (int i = 0; i < cost_vec.size(); i++)
  {
    cost_vec[i] = cost_vec.size() - i;
  }
  std::cout << std::endl;
  float* norm_d;
  float* costs_d;
  float sum_cost_compute;
  HANDLE_ERROR(cudaMalloc((void**)&norm_d, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMemcpy(costs_d, cost_vec.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));
  computeBaselineCostKernel<num_rollouts><<<1, 1024>>>(costs_d, norm_d);
  HANDLE_ERROR(cudaMemcpy(&sum_cost_compute, norm_d, sizeof(float), cudaMemcpyDeviceToHost));

  float sum_cost_known = *std::min_element(cost_vec.begin(), cost_vec.end());
  ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}

TEST_F(NormExpKernel, computeExpNorm_Test)
{
  const int num_rollouts = 555;
  std::array<float, num_rollouts> cost_vec = { 0 };
  std::array<float, num_rollouts> normalized_compute = { 0 };
  std::array<float, num_rollouts> normalized_known = { 0 };
  float gamma = 0.3;

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float baseline = *std::min_element(cost_vec.begin(), cost_vec.end());

  for (int i = 0; i < num_rollouts; i++)
  {
    normalized_known[i] = expf(-gamma * (cost_vec[i] - baseline));
  }

  launchNormExp_KernelTest<num_rollouts>(cost_vec, gamma, baseline, normalized_compute);

  array_assert_float_eq<num_rollouts>(normalized_compute, normalized_known);
}

TEST_F(NormExpKernel, MinMaxWeightKernelComputesWeightsAndEss)
{
  constexpr int num_rollouts = 4;
  std::array<float, num_rollouts> costs{10.0F, 20.0F, 30.0F, 40.0F};
  std::array<float, num_rollouts> weights{};
  float* costs_d = nullptr;
  mppi::kernels::CostWeightStats* stats_d = nullptr;
  mppi::kernels::CostWeightStats stats;
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&stats_d, sizeof(stats)));
  HANDLE_ERROR(cudaMemcpy(costs_d, costs.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));

  mppi::kernels::launchMinMaxWeightKernel(
      num_rollouts, 64, costs_d, nullptr, 1.0F, 0.95F, 1.0E-6F, stats_d, nullptr, true);
  HANDLE_ERROR(cudaMemcpy(weights.data(), costs_d, sizeof(float) * num_rollouts, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&stats, stats_d, sizeof(stats), cudaMemcpyDeviceToHost));

  float expected_sum = 0.0F;
  float expected_squared_sum = 0.0F;
  for (int i = 0; i < num_rollouts; ++i)
  {
    const float expected_weight = expf(-static_cast<float>(i) / 3.0F);
    expected_sum += expected_weight;
    expected_squared_sum += expected_weight * expected_weight;
  }
  for (int i = 0; i < num_rollouts; ++i)
  {
    const float expected_weight = expf(-static_cast<float>(i) / 3.0F) / expected_sum;
    EXPECT_NEAR(weights[i], expected_weight, 1.0E-6F);
  }
  EXPECT_FLOAT_EQ(stats.min_cost, 10.0F);
  EXPECT_FLOAT_EQ(stats.max_cost, 40.0F);
  EXPECT_NEAR(stats.normalization_upper_cost, 40.0F, 1.0E-4F);
  EXPECT_FLOAT_EQ(stats.rollout_min_cost, 10.0F);
  EXPECT_NEAR(stats.normalizer, expected_sum, 1.0E-6F);
  EXPECT_NEAR(stats.effective_sample_size, expected_sum * expected_sum / expected_squared_sum, 1.0E-5F);
  EXPECT_FLOAT_EQ(stats.raw_cost_sum, 100.0F);
  EXPECT_FLOAT_EQ(stats.raw_cost_squared_sum, 3000.0F);

  HANDLE_ERROR(cudaFree(stats_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

TEST_F(NormExpKernel, MinMaxWeightKernelHandlesDegenerateCostRange)
{
  constexpr int num_rollouts = 32;
  std::array<float, num_rollouts> costs{};
  for (int i = 0; i < num_rollouts; ++i)
  {
    costs[i] = (i % 2 == 0) ? 0.0F : 5.0E-7F;
  }
  float* costs_d = nullptr;
  mppi::kernels::CostWeightStats* stats_d = nullptr;
  mppi::kernels::CostWeightStats stats;
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&stats_d, sizeof(stats)));
  HANDLE_ERROR(cudaMemcpy(costs_d, costs.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));

  mppi::kernels::launchMinMaxWeightKernel(
      num_rollouts, 64, costs_d, nullptr, 10.0F, 0.95F, 1.0E-6F, stats_d, nullptr, true);
  HANDLE_ERROR(cudaMemcpy(costs.data(), costs_d, sizeof(float) * num_rollouts, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&stats, stats_d, sizeof(stats), cudaMemcpyDeviceToHost));

  for (const float weight : costs)
  {
    EXPECT_FLOAT_EQ(weight, 1.0F / static_cast<float>(num_rollouts));
  }
  EXPECT_FLOAT_EQ(stats.normalizer, static_cast<float>(num_rollouts));
  EXPECT_FLOAT_EQ(stats.effective_sample_size, static_cast<float>(num_rollouts));
  EXPECT_FLOAT_EQ(stats.min_cost, 0.0F);
  EXPECT_FLOAT_EQ(stats.max_cost, 5.0E-7F);
  EXPECT_FLOAT_EQ(stats.rollout_min_cost, 0.0F);
  EXPECT_NEAR(stats.raw_cost_sum, 8.0E-6F, 1.0E-12F);
  EXPECT_NEAR(stats.raw_cost_squared_sum, 4.0E-12F, 1.0E-18F);

  HANDLE_ERROR(cudaFree(stats_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

TEST_F(NormExpKernel, MinMaxWeightKernelDoesNotRewardNonFiniteCosts)
{
  constexpr int num_rollouts = 4;
  std::array<float, num_rollouts> costs{
      7.0F, 7.0F, std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  float* costs_d = nullptr;
  mppi::kernels::CostWeightStats* stats_d = nullptr;
  mppi::kernels::CostWeightStats stats;
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&stats_d, sizeof(stats)));
  HANDLE_ERROR(cudaMemcpy(costs_d, costs.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));

  mppi::kernels::launchMinMaxWeightKernel(
      num_rollouts, 64, costs_d, nullptr, 1.0F, 0.95F, 1.0E-6F, stats_d, nullptr, true);
  HANDLE_ERROR(cudaMemcpy(costs.data(), costs_d, sizeof(float) * num_rollouts, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&stats, stats_d, sizeof(stats), cudaMemcpyDeviceToHost));

  EXPECT_GT(costs[0], costs[2]);
  EXPECT_GT(costs[1], costs[3]);
  EXPECT_NEAR(std::accumulate(costs.begin(), costs.end(), 0.0F), 1.0F, 1.0E-6F);
  EXPECT_TRUE(std::isfinite(stats.effective_sample_size));
  EXPECT_FLOAT_EQ(stats.raw_cost_sum, 14.0F);
  EXPECT_FLOAT_EQ(stats.raw_cost_squared_sum, 98.0F);
  EXPECT_FLOAT_EQ(stats.rollout_min_cost, 7.0F);

  HANDLE_ERROR(cudaFree(stats_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

TEST_F(NormExpKernel, RobustPercentilePreventsSingleOutlierFromDefiningScale)
{
  constexpr int num_rollouts = 101;
  std::array<float, num_rollouts> costs{};
  for (int i = 0; i < num_rollouts - 1; ++i)
  {
    costs[i] = static_cast<float>(i);
  }
  costs.back() = 1.0E6F;

  float* costs_d = nullptr;
  mppi::kernels::CostWeightStats* stats_d = nullptr;
  mppi::kernels::CostWeightStats stats;
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&stats_d, sizeof(stats)));
  HANDLE_ERROR(cudaMemcpy(costs_d, costs.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));

  mppi::kernels::launchMinMaxWeightKernel(
      num_rollouts, 256, costs_d, nullptr, 1.0F, 0.95F, 1.0E-6F, stats_d, nullptr, true);
  HANDLE_ERROR(cudaMemcpy(&stats, stats_d, sizeof(stats), cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(stats.min_cost, 0.0F);
  EXPECT_FLOAT_EQ(stats.max_cost, 1.0E6F);
  EXPECT_NEAR(stats.normalization_upper_cost, 95.0F, 0.1F);

  HANDLE_ERROR(cudaFree(stats_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

TEST_F(NormExpKernel, ReportsUnsafeRolloutFraction)
{
  constexpr int num_rollouts = 10;
  std::array<float, num_rollouts> costs{};
  std::array<int, num_rollouts> crash_status{0, 1, 0, 0, 1, 0, 0, 0, 1, 0};
  for (int i = 0; i < num_rollouts; ++i)
  {
    costs[i] = static_cast<float>(i);
  }

  float* costs_d = nullptr;
  int* crash_status_d = nullptr;
  mppi::kernels::CostWeightStats* stats_d = nullptr;
  mppi::kernels::CostWeightStats stats;
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&crash_status_d, sizeof(int) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&stats_d, sizeof(stats)));
  HANDLE_ERROR(cudaMemcpy(costs_d, costs.data(), sizeof(float) * num_rollouts, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(
      crash_status_d, crash_status.data(), sizeof(int) * num_rollouts, cudaMemcpyHostToDevice));

  mppi::kernels::launchMinMaxWeightKernel(
      num_rollouts, 64, costs_d, crash_status_d, 1.0F, 0.95F, 1.0E-6F, stats_d, nullptr, true);
  HANDLE_ERROR(cudaMemcpy(&stats, stats_d, sizeof(stats), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(stats.unsafe_rollout_fraction, 0.3F, 1.0E-6F);

  HANDLE_ERROR(cudaFree(stats_d));
  HANDLE_ERROR(cudaFree(crash_status_d));
  HANDLE_ERROR(cudaFree(costs_d));
}

namespace
{
struct WeightResult
{
  std::vector<float> weights;
  mppi::kernels::CostWeightStats stats;
};

WeightResult safeWeights(const std::vector<float>& costs, const std::vector<int>& unsafe, float lambda = 2.0F,
                         float percentile = 0.95F)
{
  WeightResult result;
  result.weights.resize(costs.size());
  float* costs_d = nullptr;
  int* unsafe_d = nullptr;
  mppi::kernels::CostWeightStats* stats_d = nullptr;
  HANDLE_ERROR(cudaMalloc((void**)&costs_d, costs.size() * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&stats_d, sizeof(result.stats)));
  HANDLE_ERROR(cudaMemcpy(costs_d, costs.data(), costs.size() * sizeof(float), cudaMemcpyHostToDevice));
  if (!unsafe.empty())
  {
    HANDLE_ERROR(cudaMalloc((void**)&unsafe_d, unsafe.size() * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(unsafe_d, unsafe.data(), unsafe.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  mppi::kernels::launchMinMaxWeightKernel(static_cast<int>(costs.size()), 256, costs_d, unsafe_d, 1.0F / lambda,
                                          percentile, 1.0E-6F, stats_d, nullptr, true);
  HANDLE_ERROR(cudaMemcpy(result.weights.data(), costs_d, costs.size() * sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&result.stats, stats_d, sizeof(result.stats), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(stats_d));
  HANDLE_ERROR(cudaFree(costs_d));
  if (unsafe_d)
    HANDLE_ERROR(cudaFree(unsafe_d));
  return result;
}
}  // namespace

TEST_F(NormExpKernel, UnsafeMajorityCannotDominateEvenAtHighLambda)
{
  std::vector<float> costs(20, 1.0F);
  std::vector<int> unsafe(20, 1);
  costs[0] = 0.0F;
  unsafe[0] = 0;
  const auto result = safeWeights(costs, unsafe);
  EXPECT_FLOAT_EQ(result.weights[0], 1.0F);
  for (std::size_t i = 1; i < costs.size(); ++i)
    EXPECT_FLOAT_EQ(result.weights[i], 0.0F);
  EXPECT_EQ(result.stats.eligible_count, 1);
  EXPECT_FLOAT_EQ(result.stats.effective_sample_size, 1.0F);
}

TEST_F(NormExpKernel, AllUnsafeOrNonFiniteCostsHaveNoEligibleWeight)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const auto& result : { safeWeights({ 1.0F, 2.0F }, { 1, 1 }), safeWeights({ nan, inf }, {}) })
  {
    EXPECT_EQ(result.stats.eligible_count, 0);
    EXPECT_FLOAT_EQ(result.stats.normalizer, 0.0F);
    EXPECT_FLOAT_EQ(result.stats.effective_sample_size, 0.0F);
    for (float weight : result.weights)
      EXPECT_FLOAT_EQ(weight, 0.0F);
  }
}

TEST_F(NormExpKernel, ExactPercentileHandlesExtremeOutliersNegativeCostsAndEndpoints)
{
  std::vector<float> costs(101);
  for (int i = 0; i < 100; ++i)
    costs[i] = static_cast<float>(i - 100);
  costs.back() = 1.0E12F;
  EXPECT_FLOAT_EQ(safeWeights(costs, {}).stats.normalization_upper_cost, -5.0F);
  EXPECT_FLOAT_EQ(safeWeights(costs, {}, 2.0F, 0.0F).stats.normalization_upper_cost, -100.0F);
  EXPECT_FLOAT_EQ(safeWeights(costs, {}, 2.0F, 1.0F).stats.normalization_upper_cost, 1.0E12F);
  // An unsafe low cost must not shift the normalization baseline of the safe population.
  const auto eligible = safeWeights({ -1000.0F, -2.0F, 2.0F }, { 1, 0, 0 }, 1.0F, 1.0F);
  EXPECT_FLOAT_EQ(eligible.stats.rollout_min_cost, -2.0F);
  EXPECT_FLOAT_EQ(eligible.weights[0], 0.0F);
  EXPECT_NEAR(eligible.weights[2] / eligible.weights[1], std::exp(-1.0F), 1.0E-6F);
}

TEST_F(NormExpKernel, FiniteExtremeRangeDoesNotOverflowNormalization)
{
  const float maximum = std::numeric_limits<float>::max();
  const auto result = safeWeights({ -maximum, 0.0F, maximum }, {}, 1.0F, 1.0F);
  EXPECT_NEAR(result.weights[1] / result.weights[0], std::exp(-0.5F), 1.0E-6F);
  EXPECT_NEAR(result.weights[2] / result.weights[0], std::exp(-1.0F), 1.0E-6F);
}

TEST_F(NormExpKernel, ZeroWeightsPreserveEveryControlIncludingTheFirst)
{
  constexpr int rollouts = 4;
  constexpr int horizon = 2;
  std::vector<float> weights(rollouts, 0.0F);
  std::vector<float> samples(rollouts * horizon * 2, std::numeric_limits<float>::quiet_NaN());
  const std::vector<float> seed{ 0.7F, 0.2F, 0.6F, 0.3F };
  std::vector<float> output(seed.size());
  float *weights_d = nullptr, *samples_d = nullptr, *mean_d = nullptr;
  HANDLE_ERROR(cudaMalloc((void**)&weights_d, weights.size() * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&samples_d, samples.size() * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&mean_d, seed.size() * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(weights_d, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(samples_d, samples.data(), samples.size() * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mean_d, seed.data(), seed.size() * sizeof(float), cudaMemcpyHostToDevice));
  mppi::kernels::launchWeightedReductionKernel<2>(weights_d, samples_d, mean_d, 1.0F, horizon, rollouts, 1, nullptr,
                                                  true);
  HANDLE_ERROR(cudaMemcpy(output.data(), mean_d, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_EQ(output, seed);
  HANDLE_ERROR(cudaFree(mean_d));
  HANDLE_ERROR(cudaFree(samples_d));
  HANDLE_ERROR(cudaFree(weights_d));
}

TEST(EssLambdaAdaptation, UsesEligiblePopulationAndBoundedLogFeedback)
{
  mppi::kernels::CostWeightStats stats;
  stats.finite_count = 10000;
  stats.eligible_count = 100;
  stats.minimum_cost_count = 1;
  stats.normalizer = 10.0F;
  stats.normalization_upper_cost = 1.0F;
  stats.effective_sample_size = 10.0F;
  const auto next = [&]() { return mppi::controllers::adaptEssLambda(0.05F, stats, 0.1F, 0.3F, 0.005F, 2.0F); };
  EXPECT_FLOAT_EQ(next(), 0.05F);  // 10% of eligible samples, not of all finite samples.
  stats.effective_sample_size = 1.0F;
  EXPECT_GT(next(), 0.09F);  // Recover much faster than the old 3%-per-cycle ceiling.
  EXPECT_LE(next(), 0.1F);
  stats.effective_sample_size = 100.0F;
  EXPECT_LT(next(), 0.03F);
  EXPECT_GE(next(), 0.025F);
  stats.unsafe_rollout_fraction = 0.99F;
  EXPECT_LT(next(), 0.05F);  // Safety flags cannot force maximum temperature.
  EXPECT_FLOAT_EQ(mppi::controllers::adaptEssLambda(0.005F, stats, 0.1F, 0.3F, 0.005F, 2.0F), 0.005F);
  stats.effective_sample_size = 1.0F;
  EXPECT_FLOAT_EQ(mppi::controllers::adaptEssLambda(1.9F, stats, 0.1F, 1.0F, 0.005F, 2.0F), 2.0F);
}

TEST(EssLambdaAdaptation, DoesNotWindUpOnFlatTiedOrFailedPopulations)
{
  mppi::kernels::CostWeightStats stats;
  stats.eligible_count = 100;
  stats.minimum_cost_count = 100;
  stats.normalizer = 100.0F;
  stats.effective_sample_size = 100.0F;
  const auto next = [&]() { return mppi::controllers::adaptEssLambda(0.05F, stats, 0.1F, 0.3F, 0.005F, 2.0F); };
  for (int cycle = 0; cycle < 100; ++cycle)
    EXPECT_FLOAT_EQ(next(), 0.05F);
  stats.normalization_upper_cost = 1.0F;
  stats.minimum_cost_count = 20;
  stats.effective_sample_size = 20.0F;
  EXPECT_FLOAT_EQ(next(), 0.05F);  // Requested ESS 10 is below the attainable floor 20.
  stats.minimum_cost_count = 1;
  stats.normalizer = 0.0F;
  stats.eligible_count = 0;
  EXPECT_FLOAT_EQ(next(), 0.05F);
}

TEST(EssLambdaAdaptation, ConvergesAcrossCyclesOnAnInformativePopulation)
{
  mppi::kernels::CostWeightStats stats;
  stats.eligible_count = 100;
  stats.minimum_cost_count = 1;
  stats.normalization_upper_cost = 1.0F;
  float lambda = 0.005F;
  for (int cycle = 0; cycle < 50; ++cycle)
  {
    double sum = 0.0, squared_sum = 0.0;
    for (int i = 0; i < 100; ++i)
    {
      const double w = std::exp(-(static_cast<double>(i) / 99.0) / lambda);
      sum += w;
      squared_sum += w * w;
    }
    stats.normalizer = static_cast<float>(sum);
    stats.effective_sample_size = static_cast<float>(sum * sum / squared_sum);
    const float updated = mppi::controllers::adaptEssLambda(lambda, stats, 0.2F, 0.3F, 0.005F, 2.0F);
    EXPECT_GE(updated, lambda / 2.0F);
    EXPECT_LE(updated, lambda * 2.0F);
    lambda = updated;
  }
  EXPECT_NEAR(stats.effective_sample_size, 20.0F, 0.1F);
}

TEST_F(NormExpKernel, comparisonTestAutorallyMPPI_Generic)
{
  const int num_rollouts = 28754;
  const int blocksize_x = 8;
  const int blocksize_y = 8;
  std::array<float, num_rollouts> cost_vec = { 0 };
  std::array<float, num_rollouts> normalized_autorally = { 0 };
  std::array<float, num_rollouts> normalized_generic = { 0 };
  float gamma = 0.3;

  // Use a range based for loop to set the cost
  for (auto& cost : cost_vec)
  {
    cost = distribution(generator);
  }

  float baseline = *std::min_element(cost_vec.begin(), cost_vec.end());

  launchGenericNormExpKernelTest<num_rollouts, blocksize_x>(cost_vec, gamma, baseline, normalized_generic);

  for (int i = 0; i < num_rollouts; i++)
  {
    float cost = cost_vec[i] - baseline;
    cost = expf(-gamma * cost);
    EXPECT_FLOAT_EQ(normalized_generic[i], cost);
  }
}

TEST_F(NormExpKernel, comparisonTestHostvsDeviceBaselineNormalizerCalculation)
{
  const int num_rollouts = 10000;
  const int blocksize_x = 8;
  const int num_iterations = 2500;
  std::array<float, num_rollouts> cost_vec = { 0 };
  std::array<float, num_rollouts> host_dev_costs = { 0 };
  std::array<float, num_rollouts> dev_only_costs = { 0 };
  float lambda = 0.3;
  double old_method_ms = 0;
  double new_method_ms = 0;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float* costs_dev_only_d;
  float* costs_host_only_d;
  float2* baseline_and_normalizer_d;
  float2 host_components, device_components;
  HANDLE_ERROR(cudaMalloc((void**)&baseline_and_normalizer_d, sizeof(float2)));
  HANDLE_ERROR(cudaMalloc((void**)&costs_dev_only_d, sizeof(float) * num_rollouts));
  HANDLE_ERROR(cudaMalloc((void**)&costs_host_only_d, sizeof(float) * num_rollouts));

  // Use a range based for loop to set the cost
  for (int i = 0; i < num_iterations; i++)
  {
    for (auto& cost : cost_vec)
    {
      cost = distribution(generator);
    }

    /**
     * @brief Prep CUDA components
     *
     */
    HANDLE_ERROR(cudaMemcpyAsync(costs_dev_only_d, cost_vec.data(), sizeof(float) * num_rollouts,
                                 cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(costs_host_only_d, cost_vec.data(), sizeof(float) * num_rollouts,
                                 cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    auto start_old_method_t = std::chrono::steady_clock::now();
    // Run old method to transform costs
    HANDLE_ERROR(cudaMemcpyAsync(host_dev_costs.data(), costs_host_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    host_components.x = mppi::kernels::computeBaselineCost(host_dev_costs.data(), num_rollouts);
    mppi::kernels::launchNormExpKernel(num_rollouts, blocksize_x, costs_host_only_d, 1.0 / lambda, host_components.x,
                                       stream, false);
    HANDLE_ERROR(cudaMemcpyAsync(host_dev_costs.data(), costs_host_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    host_components.y = mppi::kernels::computeNormalizer(host_dev_costs.data(), num_rollouts);
    old_method_ms += (std::chrono::steady_clock::now() - start_old_method_t).count() / 1e6;

    auto start_new_method_t = std::chrono::steady_clock::now();
    // Run new method to transform costs
    mppi::kernels::launchWeightTransformKernel<num_rollouts>(costs_dev_only_d, baseline_and_normalizer_d, 1.0 / lambda,
                                                             1, stream, false);
    HANDLE_ERROR(cudaMemcpyAsync(dev_only_costs.data(), costs_dev_only_d, num_rollouts * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(
        cudaMemcpyAsync(&device_components, baseline_and_normalizer_d, sizeof(float2), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    new_method_ms += (std::chrono::steady_clock::now() - start_new_method_t).count() / 1e6;
  }

  std::cout << "Old method averaged " << old_method_ms / num_iterations << " ms and the new method averaged "
            << new_method_ms / num_iterations << " ms" << std::endl;

  for (int i = 0; i < num_rollouts; i++)
  {
    ASSERT_FLOAT_EQ(dev_only_costs[i], host_dev_costs[i]);
  }
  ASSERT_FLOAT_EQ(device_components.x, host_components.x);
  ASSERT_FLOAT_EQ(device_components.y, host_components.y);
}

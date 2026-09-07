#include <atomic>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/core/mppi_common.cuh>
#include <mppi/utils/nvtx.cuh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#define VANILLA_MPPI_TEMPLATE                                                                                          \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T>

#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

VANILLA_MPPI_TEMPLATE
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt,
                                   int max_iter, float lambda, float alpha, int num_timesteps,
                                   const Eigen::Ref<const control_trajectory>& init_control_traj, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, init_control_traj,
                 stream)
{
  try
  {
    // Allocate CUDA memory for the controller
    allocateCUDAMemory();

    // Copy the noise std_dev to the device
    // this->copyControlStdDevToDevice();

    chooseAppropriateKernel();
  }
  catch (...)
  {
    releaseWeightBuffers();
    throw;
  }
}

VANILLA_MPPI_TEMPLATE
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler,
                                   PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
{
  try
  {
    // Allocate CUDA memory for the controller
    allocateCUDAMemory();

    // // Copy the noise std_dev to the device
    // this->copyControlStdDevToDevice();
    chooseAppropriateKernel();
  }
  catch (...)
  {
    releaseWeightBuffers();
    throw;
  }
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::chooseAppropriateKernel()
{
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
  unsigned single_kernel_byte_size = mppi::kernels::calcRolloutCombinedKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_dyn_kernel_byte_size = mppi::kernels::calcRolloutDynamicsKernelSharedMemSize(
      this->model_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_cost_kernel_byte_size =
      mppi::kernels::calcRolloutCostKernelSharedMemSize(this->cost_, this->sampler_, this->params_.cost_rollout_dim_);
  unsigned vis_single_kernel_byte_size = mppi::kernels::calcVisualizeKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->getNumTimesteps(), this->params_.visualize_dim_);

  bool too_much_mem_single_kernel = single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_vis_kernel = vis_single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_split_kernel = split_dyn_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_split_kernel = too_much_mem_split_kernel || split_cost_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_single_kernel = too_much_mem_single_kernel || too_much_mem_vis_kernel;

  if (too_much_mem_split_kernel && too_much_mem_single_kernel)
  {
    std::string error_msg =
        "There is not enough shared memory on the GPU for either rollout kernel option. The combined rollout kernel "
        "takes " +
        std::to_string(single_kernel_byte_size) + " bytes, the cost rollout kernel takes " +
        std::to_string(split_cost_kernel_byte_size) + " bytes, the dynamics rollout kernel takes " +
        std::to_string(split_dyn_kernel_byte_size) + " bytes, the combined visualization kernel takes " +
        std::to_string(vis_single_kernel_byte_size) + " bytes, and the max is " +
        std::to_string(deviceProp.sharedMemPerBlock) +
        " bytes. Considering lowering the corresponding thread block sizes.";
    throw std::runtime_error(error_msg);
  }
  else if (too_much_mem_single_kernel)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    return;
  }
  else if (too_much_mem_split_kernel)
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    return;
  }

  // Send the nominal control to the device
  this->copyNominalControlToDevice(false);
  state_array zero_state = this->model_->getZeroState();
  // Send zero state to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, zero_state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));
  // Generate noise data
  this->sampler_->generateSamples(1, 0, this->gen_, true);

  float single_kernel_time_ms = std::numeric_limits<float>::infinity();
  float split_kernel_time_ms = std::numeric_limits<float>::infinity();

  // Evaluate each kernel that is applicable
  auto start_single_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_single_kernel; i++)
  {
    mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
        this->params_.dynamics_rollout_dim_, this->stream_, true, rollout_crash_status_d_);
  }
  auto end_single_kernel_time = std::chrono::steady_clock::now();
  auto start_split_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_split_kernel; i++)
  {
    mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
        this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_, this->stream_, true,
        rollout_crash_status_d_);
  }
  auto end_split_kernel_time = std::chrono::steady_clock::now();

  // calc times
  if (!too_much_mem_single_kernel)
  {
    single_kernel_time_ms = mppi::math::timeDiffms(end_single_kernel_time, start_single_kernel_time);
  }
  if (!too_much_mem_split_kernel)
  {
    split_kernel_time_ms = mppi::math::timeDiffms(end_split_kernel_time, start_split_kernel_time);
  }
  std::string kernel_choice = "";
  if (split_kernel_time_ms < single_kernel_time_ms)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    kernel_choice = "split ";
  }
  else
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    kernel_choice = "single";
  }
  this->logger_->info("Choosing %s kernel based on split taking %f ms and single taking %f ms after %d iterations\n",
                      kernel_choice.c_str(), split_kernel_time_ms, single_kernel_time_ms,
                      this->getNumKernelEvaluations());
}

VANILLA_MPPI_TEMPLATE
VanillaMPPI::~VanillaMPPIController()
{
  releaseWeightBuffers();
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::releaseWeightBuffers() noexcept
{
  if (weight_stats_d_ || weight_stats_h_ || rollout_crash_status_d_)
    gpuAssert(cudaStreamSynchronize(this->stream_), __FILE__, __LINE__, false);
  cudaFreeNoThrow(weight_stats_d_);
  if (weight_stats_h_ != nullptr)
  {
    gpuAssert(cudaFreeHost(weight_stats_h_), __FILE__, __LINE__, false);
    weight_stats_h_ = nullptr;
  }
  cudaFreeNoThrow(rollout_crash_status_d_);
  weight_stats_capacity_ = 0;
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::configureEssLambdaAdaptation(float target_ess_ratio, float adaptation_gain, float lambda_min,
                                               float lambda_max, float unsafe_rollout_fraction_threshold,
                                               float cost_normalization_percentile)
{
  target_ess_ratio_ = std::max(0.0F, std::min(1.0F, target_ess_ratio));
  lambda_adaptation_gain_ = std::max(0.0F, adaptation_gain);
  lambda_min_ = std::max(1.0E-6F, lambda_min);
  lambda_max_ = std::max(lambda_min_, lambda_max);
  unsafe_rollout_fraction_threshold_ = std::max(0.0F, std::min(1.0F, unsafe_rollout_fraction_threshold));
  cost_normalization_percentile_ = std::max(0.0F, std::min(1.0F, cost_normalization_percentile));
  lambda_adaptation_enabled_ = true;
  last_weight_lambda_ = std::max(lambda_min_, std::min(lambda_max_, this->getLambda()));
  last_iteration_weight_lambda_ = last_weight_lambda_;
  this->setLambda(last_weight_lambda_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::downloadImportanceWeightsToHost()
{
  HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_, NUM_ROLLOUTS * sizeof(float),
                               cudaMemcpyDeviceToHost, this->stream_));
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost();

  const int num_iterations = std::max(0, this->getNumIters());
  ensureWeightStatsCapacity(std::max(1, num_iterations));

  // Send the initial condition to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));

  const float rollout_count = std::max(static_cast<float>(NUM_ROLLOUTS), 1.0F);
  float baseline_prev = std::isfinite(this->free_energy_statistics_.real_sys.previousBaseline) ?
                            this->free_energy_statistics_.real_sys.previousBaseline :
                            0.0F;
  iteration_effective_sample_sizes_.clear();
  iteration_effective_sample_sizes_.reserve(static_cast<std::size_t>(num_iterations));
  // Apply the temperature prepared by the previous control step before entering the optimization
  // loop; it then remains fixed for every iteration in this step.
  if (lambda_adaptation_enabled_)
  {
    last_weight_lambda_ = std::max(lambda_min_, std::min(lambda_max_, last_weight_lambda_));
  }
  else
  {
    last_weight_lambda_ = std::max(1.0E-6F, this->getLambda());
  }
  last_iteration_weight_lambda_ = last_weight_lambda_;
  this->setLambda(last_weight_lambda_);

  // The host-provided warm start is needed only by iteration zero. Every subsequent iteration
  // consumes the control mean produced by the preceding device reduction.
  if (num_iterations > 0)
  {
    this->copyNominalControlToDevice(false);
  }

  for (int opt_iter = 0; opt_iter < num_iterations; opt_iter++)
  {
    // Generate noise data
    {
      mppi::instrumentation::ScopedNvtxRange range("MPPI/random_sampling", mppi::instrumentation::NvtxColor::SAMPLING);
      this->sampler_->generateSamples(optimization_stride, opt_iter, this->gen_, false);
    }

    // Launch the rollout kernel
    {
      mppi::instrumentation::ScopedNvtxRange range("MPPI/rollout", mppi::instrumentation::NvtxColor::ROLLOUT);
      if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS)
      {
        mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
            this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
            this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
            this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_, this->stream_, false,
            rollout_crash_status_d_);
      }
      else if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
      {
        mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
            this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
            this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
            this->params_.dynamics_rollout_dim_, this->stream_, false, rollout_crash_status_d_);
      }
    }

    // Preserve raw rollout costs only for explicitly enabled per-iteration diagnostics.
    if (requiresRawRolloutCostsForIteration())
    {
      mppi::instrumentation::ScopedNvtxRange range("MPPI/capture_iteration_debug",
                                                   mppi::instrumentation::NvtxColor::DEBUG);
      HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                   NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
      HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
      optimizationIterationComplete(opt_iter);
    }

    // One device kernel computes robust-normalized exponential weights, the weight normalizer,
    // ESS, and safety statistics. The per-iteration records remain on-device until the loop ends.
    {
      mppi::instrumentation::ScopedNvtxRange range("MPPI/weight_statistics",
                                                   mppi::instrumentation::NvtxColor::STATISTICS);
      mppi::kernels::launchMinMaxWeightKernel(NUM_ROLLOUTS, this->getNormExpThreads(), this->trajectory_costs_d_,
                                              rollout_crash_status_d_, 1.0F / last_iteration_weight_lambda_,
                                              cost_normalization_percentile_, 1.0E-6F, weight_stats_d_ + opt_iter,
                                              this->stream_, false);
    }
    // The fused kernel normalizes weights in-place; sampler reduction therefore uses unity.
    this->setNormalizer(1.0F);

    {
      mppi::instrumentation::ScopedNvtxRange range("MPPI/weighted_control_reduction",
                                                   mppi::instrumentation::NvtxColor::REDUCTION);
      this->sampler_->updateDistributionParamsFromDeviceOnly(this->trajectory_costs_d_, 1.0F, 0, false);
    }
  }

  if (num_iterations > 0)
  {
    mppi::instrumentation::ScopedNvtxRange range("MPPI/final_result_download",
                                                 mppi::instrumentation::NvtxColor::DATA_TRANSFER);
    // Queue both final results before the sampler performs the control step's only required host
    // synchronization. The optimization loop above contains no host transfers or barriers unless
    // explicit per-iteration raw-rollout diagnostics are enabled.
    {
      mppi::instrumentation::ScopedNvtxRange enqueue_range(
          "MPPI/final_result_enqueue_weight_stats", mppi::instrumentation::NvtxColor::DATA_TRANSFER);
      HANDLE_ERROR(cudaMemcpyAsync(weight_stats_h_, weight_stats_d_,
                                   static_cast<std::size_t>(num_iterations) * sizeof(*weight_stats_h_),
                                   cudaMemcpyDeviceToHost, this->stream_));
    }
    this->sampler_->setHostOptimalControlSequence(this->control_.data(), 0, true);
    device_optimal_control_ = this->control_;
    last_weight_stats_index_ = num_iterations - 1;
  }
  else
  {
    device_optimal_control_ = this->control_;
  }

  for (int opt_iter = 0; opt_iter < num_iterations; ++opt_iter)
  {
    const auto& weight_stats = weight_stats_h_[opt_iter];

    // Keep the controller baseline absolute and current, but protect against all-failed rollouts.
    if (weight_stats.rollout_min_cost < FLT_MAX)
    {
      this->setBaseline(weight_stats.rollout_min_cost);
    }
    iteration_effective_sample_sizes_.push_back(weight_stats.effective_sample_size);

    if (this->getBaselineCost() > baseline_prev + 1.0F)
    {
      this->logger_->debug("Previous Baseline: %f\n         Baseline: %f\n", baseline_prev, this->getBaselineCost());
    }
    baseline_prev = this->getBaselineCost();

    const float mean_weight = weight_stats.normalizer / rollout_count;
    this->free_energy_statistics_.real_sys.freeEnergyMean =
        -last_iteration_weight_lambda_ * std::log(std::max(mean_weight, 1.0E-12F));

    const float mean_raw_cost = weight_stats.raw_cost_sum / rollout_count;
    const float mean_squared_raw_cost = weight_stats.raw_cost_squared_sum / rollout_count;
    this->free_energy_statistics_.real_sys.freeEnergyVariance =
        std::max(0.0F, mean_squared_raw_cost - mean_raw_cost * mean_raw_cost);
    const float variance_scale = this->free_energy_statistics_.real_sys.freeEnergyVariance /
                                 std::max(std::abs(mean_raw_cost) * std::sqrt(rollout_count), 1.0E-12F);
    this->free_energy_statistics_.real_sys.freeEnergyModifiedVariance =
        last_iteration_weight_lambda_ * (variance_scale + 0.5F * variance_scale * variance_scale);

    // Adapt only from the converged iteration, and retain the result for the next control step.
    // Keep params_.lambda_ unchanged so every iteration and post-step visualization in this
    // control cycle uses the same temperature.
    if (lambda_adaptation_enabled_ && opt_iter == num_iterations - 1 && !iteration_effective_sample_sizes_.empty())
    {
      if (weight_stats.unsafe_rollout_fraction >= unsafe_rollout_fraction_threshold_)
      {
        // Panic melt only when the rollout population itself reports unavoidable safety contact.
        last_weight_lambda_ = lambda_max_;
      }
      else
      {
        const float ess_ratio = iteration_effective_sample_sizes_.back() / rollout_count;
        const float error = target_ess_ratio_ - ess_ratio;
        const float lambda_multiplier = std::exp(lambda_adaptation_gain_ * error);
        last_weight_lambda_ = std::max(lambda_min_, std::min(lambda_max_, last_weight_lambda_ * lambda_multiplier));
      }
    }
  }

  this->free_energy_statistics_.real_sys.normalizerPercent =
      iteration_effective_sample_sizes_.empty() ? 0.0F : iteration_effective_sample_sizes_.back() / rollout_count;
  this->free_energy_statistics_.real_sys.increase =
      this->getBaselineCost() - this->free_energy_statistics_.real_sys.previousBaseline;
  smoothControlTrajectory();
  computeStateTrajectory(state);
  state_array zero_state = this->model_->getZeroState();
  for (int i = 0; i < this->getNumTimesteps(); i++)
  {
    this->model_->enforceConstraints(zero_state, this->control_.col(i));
  }

  // Copy back sampled trajectories
  {
    mppi::instrumentation::ScopedNvtxRange range("MPPI/stage_visualization_buffers",
                                                 mppi::instrumentation::NvtxColor::DEBUG);
    this->copySampledControlFromDevice(false);
    if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL && this->getTotalSampledTrajectories() > 0)
    {  // copy initial state to vis initial state for use with visualizeKernel
      HANDLE_ERROR(cudaMemcpyAsync(this->vis_initial_state_d_, this->initial_state_d_, sizeof(float) * DYN_T::STATE_DIM,
                                   cudaMemcpyDeviceToDevice, this->vis_stream_));
    }
    if (this->num_top_control_trajectories_ > 0)
    {
      downloadImportanceWeightsToHost();
    }
    this->copyTopControlFromDevice(true);
  }
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::allocateCUDAMemory()
{
  PARENT_CLASS::allocateCUDAMemoryHelper();
  ensureWeightStatsCapacity(std::max(1, this->getNumIters()));
  HANDLE_ERROR(cudaMalloc((void**)&rollout_crash_status_d_, NUM_ROLLOUTS * sizeof(int)));
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::ensureWeightStatsCapacity(int required_capacity)
{
  required_capacity = std::max(1, required_capacity);
  if (weight_stats_capacity_ >= required_capacity)
  {
    return;
  }
  if (weight_stats_d_ != nullptr)
  {
    HANDLE_ERROR(cudaFree(weight_stats_d_));
    weight_stats_d_ = nullptr;
  }
  if (weight_stats_h_ != nullptr)
  {
    HANDLE_ERROR(cudaFreeHost(weight_stats_h_));
    weight_stats_h_ = nullptr;
  }
  weight_stats_capacity_ = 0;
  HANDLE_ERROR(
      cudaMalloc((void**)&weight_stats_d_, static_cast<std::size_t>(required_capacity) * sizeof(*weight_stats_d_)));
  HANDLE_ERROR(
      cudaMallocHost((void**)&weight_stats_h_, static_cast<std::size_t>(required_capacity) * sizeof(*weight_stats_h_)));
  weight_stats_capacity_ = required_capacity;
  last_weight_stats_index_ = 0;
  std::fill_n(weight_stats_h_, weight_stats_capacity_, mppi::kernels::CostWeightStats{});
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0)
{
  this->computeOutputTrajectoryHelper(this->output_, this->state_, x0, this->control_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::smoothControlTrajectory()
{
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::launchSampledVisTrajectories()
{
  const int num_sampled_trajectories = this->getTotalSampledTrajectories();

  if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS)
  {
    mppi::kernels::launchVisualizeCostKernel<COST_T, SAMPLING_T>(
        this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->sampled_outputs_d_, this->sampled_crash_status_d_,
        this->sampled_costs_d_, this->params_.cost_rollout_dim_, this->stream_, false);
  }
  else if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
  {
    mppi::kernels::launchVisualizeKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->vis_initial_state_d_, this->sampled_outputs_d_,
        this->sampled_costs_d_, this->sampled_crash_status_d_, this->params_.visualize_dim_, this->stream_, false);
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::calculateSampledStateTrajectories()
{
  mppi::instrumentation::ScopedNvtxRange range("MPPI/build_sampled_trajectories",
                                               mppi::instrumentation::NvtxColor::DEBUG);
  launchSampledVisTrajectories();
  this->downloadSampledVisTrajectoriesToHost();
}

#undef VANILLA_MPPI_TEMPLATE
#undef VanillaMPPI

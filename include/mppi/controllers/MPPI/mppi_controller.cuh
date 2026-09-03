/**
 * Created by jason on 10/30/19.
 * Creates the API for interfacing with an MPPI controller
 * should define a compute_control based on state as well
 * as return timing info
 **/

#ifndef MPPIGENERIC_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_MPPI_CONTROLLER_CUH

#include <mppi/controllers/controller.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <vector>

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
          class SAMPLING_T = ::mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>,
          class PARAMS_T = ControllerParams<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, MAX_TIMESTEPS>>
class VanillaMPPIController : public Controller<DYN_T, COST_T, FB_T, SAMPLING_T, MAX_TIMESTEPS, NUM_ROLLOUTS, PARAMS_T>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // nAeed control_array = ... so that we can initialize
  // Eigen::Matrix with Eigen::Matrix::Zero();
  typedef Controller<DYN_T, COST_T, FB_T, SAMPLING_T, MAX_TIMESTEPS, NUM_ROLLOUTS, PARAMS_T> PARENT_CLASS;
  using control_array = typename PARENT_CLASS::control_array;
  using control_trajectory = typename PARENT_CLASS::control_trajectory;
  using state_trajectory = typename PARENT_CLASS::state_trajectory;
  using state_array = typename PARENT_CLASS::state_array;
  using sampled_cost_traj = typename PARENT_CLASS::sampled_cost_traj;
  using FEEDBACK_GPU = typename PARENT_CLASS::TEMPLATED_FEEDBACK_GPU;

  /**
   *
   * Public member functions
   */
  // Constructor
  VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt, int max_iter,
                        float lambda, float alpha, int num_timesteps = MAX_TIMESTEPS,
                        const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                        cudaStream_t stream = nullptr);
  VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, PARAMS_T& params,
                        cudaStream_t stream = nullptr);

  // Destructor
  ~VanillaMPPIController();

  std::string getControllerName() const override
  {
    return "Vanilla MPPI";
  };

  /**
   * computes a new control sequence
   * @param state starting position
   */
  void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride = 1) override;

  void setPercentageSampledControlTrajectories(float new_perc)
  {
    this->setPercentageSampledControlTrajectoriesHelper(new_perc, 1);
  }

  void calculateSampledStateTrajectories() override;

  /** Run visualization rollouts on GPU only (no D2H). Pair with viz::fillRolloutsFromDevice. */
  void launchSampledVisTrajectories();

  void chooseAppropriateKernel() override;

  /** Configure robust weighting, the ESS target, and bounded temperature feedback. */
  void configureEssLambdaAdaptation(float target_ess_ratio, float adaptation_gain,
                                    float lambda_min, float lambda_max,
                                    float unsafe_rollout_fraction_threshold = 0.95F,
                                    float cost_normalization_percentile = 0.95F);

  const std::vector<float>& getIterationEffectiveSampleSizes() const
  {
    return iteration_effective_sample_sizes_;
  }

  float getLastWeightLambda() const
  {
    return last_iteration_weight_lambda_;
  }

  float getNextWeightLambda() const
  {
    return last_weight_lambda_;
  }

  float getLastMinRolloutCost() const
  {
    return lastWeightStats().min_cost;
  }

  float getLastMaxRolloutCost() const
  {
    return lastWeightStats().max_cost;
  }

  float getLastNormalizationUpperCost() const
  {
    return lastWeightStats().normalization_upper_cost;
  }

  float getLastUnsafeRolloutFraction() const
  {
    return lastWeightStats().unsafe_rollout_fraction;
  }

  float getLastUnnormalizedWeightSum() const
  {
    return lastWeightStats().normalizer;
  }

  /** Explicitly download the final normalized importance weights for debug consumers. */
  void downloadImportanceWeightsToHost();

protected:
  /**
   * Called after a rollout batch only when requiresRawRolloutCostsForIteration() is true. Derived
   * controllers can capture per-iteration diagnostics before raw costs are replaced by weights.
   */
  virtual void optimizationIterationComplete(int iteration)
  {
    (void)iteration;
  }

  /** Whether optimizationIterationComplete needs the raw rollout costs in trajectory_costs_. */
  virtual bool requiresRawRolloutCostsForIteration() const
  {
    return false;
  }

  /** Device-reduced control before the generic host smoothing/constraint pass. */
  const control_trajectory& getDeviceOptimalControlSequence() const
  {
    return device_optimal_control_;
  }

  void computeStateTrajectory(const Eigen::Ref<const state_array>& x0);

  void smoothControlTrajectory();

private:
  // ======== MUST BE OVERWRITTEN =========
  void allocateCUDAMemory();
  // ======== END MUST BE OVERWRITTEN =====

  void ensureWeightStatsCapacity(int required_capacity);

  const mppi::kernels::CostWeightStats& lastWeightStats() const
  {
    return weight_stats_h_[last_weight_stats_index_];
  }

  mppi::kernels::CostWeightStats* weight_stats_d_ = nullptr;
  /** One collision/safety flag per rollout, produced by the rollout cost kernel. */
  int* rollout_crash_status_d_ = nullptr;
  /** Pinned host mirror downloaded once after all optimization iterations complete. */
  mppi::kernels::CostWeightStats* weight_stats_h_ = nullptr;
  int weight_stats_capacity_ = 0;
  int last_weight_stats_index_ = 0;
  /** Host snapshot of the final device mean before generic host-side post-processing. */
  control_trajectory device_optimal_control_ = control_trajectory::Zero();
  std::vector<float> iteration_effective_sample_sizes_;
  /** Persisted adaptive state to be used by the next control step. */
  float last_weight_lambda_ = 1.0F;
  /** Lambda that produced the final iteration's stored weights and statistics. */
  float last_iteration_weight_lambda_ = 1.0F;
  float target_ess_ratio_ = 0.2F;
  float lambda_adaptation_gain_ = 0.1F;
  float lambda_min_ = 0.01F;
  float lambda_max_ = 2.0F;
  float unsafe_rollout_fraction_threshold_ = 0.95F;
  float cost_normalization_percentile_ = 0.95F;
  bool lambda_adaptation_enabled_ = false;
};

#if __CUDACC__
#include "mppi_controller.cu"
#endif

#endif  // MPPIGENERIC_MPPI_CONTROLLER_CUH


// Code generated by stanc v2.27.0
#include <stan/model/model_header.hpp>
namespace linear_hierarchical_model_model_namespace {

using stan::io::dump;
using stan::model::assign;
using stan::model::index_uni;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::model_base_crtp;
using stan::model::rvalue;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 30> locations_array__ = 
{" (found before start of program)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 14, column 4 to column 29)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 15, column 4 to column 34)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 16, column 4 to column 26)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 17, column 4 to column 31)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 18, column 4 to column 29)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 23, column 4 to column 29)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 24, column 4 to column 32)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 27, column 4 to column 63)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 29, column 4 to column 17)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 30, column 4 to column 11)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 33, column 8 to column 74)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 34, column 8 to column 131)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 35, column 8 to column 40)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 32, column 19 to line 36, column 5)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 32, column 4 to line 36, column 5)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 5, column 4 to column 28)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 6, column 4 to column 28)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 7, column 33 to column 34)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 7, column 4 to column 36)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 8, column 11 to column 12)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 8, column 4 to column 34)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 9, column 11 to column 12)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 9, column 4 to column 34)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 10, column 20 to column 21)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 10, column 4 to column 34)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 11, column 20 to column 21)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 11, column 4 to column 34)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 17, column 28 to column 29)",
 " (in '/Users/claytonroberts/Documents/TROPOMI_LHM/src/models/linear_hierarchical_model.stan', line 18, column 20 to column 21)"};



class linear_hierarchical_model_model final : public model_base_crtp<linear_hierarchical_model_model> {

 private:
  int M;
  int D;
  std::vector<int> day_id;
  Eigen::Matrix<double, -1, 1> NO2_obs__;
  Eigen::Matrix<double, -1, 1> CH4_obs__;
  Eigen::Matrix<double, -1, 1> sigma_N__;
  Eigen::Matrix<double, -1, 1> sigma_C__; 
  Eigen::Map<Eigen::Matrix<double, -1, 1>> NO2_obs{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double, -1, 1>> CH4_obs{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double, -1, 1>> sigma_N{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double, -1, 1>> sigma_C{nullptr, 0};
 
 public:
  ~linear_hierarchical_model_model() { }
  
  inline std::string model_name() const final { return "linear_hierarchical_model_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.27.0", "stancflags = "};
  }
  
  
  linear_hierarchical_model_model(stan::io::var_context& context__,
                                  unsigned int random_seed__ = 0,
                                  std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "linear_hierarchical_model_model_namespace::linear_hierarchical_model_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      current_statement__ = 16;
      context__.validate_dims("data initialization","M","int",
           std::vector<size_t>{});
      M = std::numeric_limits<int>::min();
      
      current_statement__ = 16;
      M = context__.vals_i("M")[(1 - 1)];
      current_statement__ = 16;
      check_greater_or_equal(function__, "M", M, 0);
      current_statement__ = 17;
      context__.validate_dims("data initialization","D","int",
           std::vector<size_t>{});
      D = std::numeric_limits<int>::min();
      
      current_statement__ = 17;
      D = context__.vals_i("D")[(1 - 1)];
      current_statement__ = 17;
      check_greater_or_equal(function__, "D", D, 0);
      current_statement__ = 18;
      validate_non_negative_index("day_id", "M", M);
      current_statement__ = 19;
      context__.validate_dims("data initialization","day_id","int",
           std::vector<size_t>{static_cast<size_t>(M)});
      day_id = std::vector<int>(M, std::numeric_limits<int>::min());
      
      current_statement__ = 19;
      day_id = context__.vals_i("day_id");
      current_statement__ = 19;
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        current_statement__ = 19;
        check_greater_or_equal(function__, "day_id[sym1__]",
                               day_id[(sym1__ - 1)], 1);
      }
      current_statement__ = 19;
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        current_statement__ = 19;
        check_less_or_equal(function__, "day_id[sym1__]",
                            day_id[(sym1__ - 1)], D);
      }
      current_statement__ = 20;
      validate_non_negative_index("NO2_obs", "M", M);
      current_statement__ = 21;
      context__.validate_dims("data initialization","NO2_obs","double",
           std::vector<size_t>{static_cast<size_t>(M)});
      NO2_obs__ = Eigen::Matrix<double, -1, 1>(M);
      new (&NO2_obs) Eigen::Map<Eigen::Matrix<double, -1, 1>>(NO2_obs__.data(), M);
      
      
      {
        std::vector<local_scalar_t__> NO2_obs_flat__;
        current_statement__ = 21;
        NO2_obs_flat__ = context__.vals_r("NO2_obs");
        current_statement__ = 21;
        pos__ = 1;
        current_statement__ = 21;
        for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
          current_statement__ = 21;
          assign(NO2_obs, NO2_obs_flat__[(pos__ - 1)],
            "assigning variable NO2_obs", index_uni(sym1__));
          current_statement__ = 21;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 22;
      validate_non_negative_index("CH4_obs", "M", M);
      current_statement__ = 23;
      context__.validate_dims("data initialization","CH4_obs","double",
           std::vector<size_t>{static_cast<size_t>(M)});
      CH4_obs__ = Eigen::Matrix<double, -1, 1>(M);
      new (&CH4_obs) Eigen::Map<Eigen::Matrix<double, -1, 1>>(CH4_obs__.data(), M);
      
      
      {
        std::vector<local_scalar_t__> CH4_obs_flat__;
        current_statement__ = 23;
        CH4_obs_flat__ = context__.vals_r("CH4_obs");
        current_statement__ = 23;
        pos__ = 1;
        current_statement__ = 23;
        for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
          current_statement__ = 23;
          assign(CH4_obs, CH4_obs_flat__[(pos__ - 1)],
            "assigning variable CH4_obs", index_uni(sym1__));
          current_statement__ = 23;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 24;
      validate_non_negative_index("sigma_N", "M", M);
      current_statement__ = 25;
      context__.validate_dims("data initialization","sigma_N","double",
           std::vector<size_t>{static_cast<size_t>(M)});
      sigma_N__ = Eigen::Matrix<double, -1, 1>(M);
      new (&sigma_N) Eigen::Map<Eigen::Matrix<double, -1, 1>>(sigma_N__.data(), M);
      
      
      {
        std::vector<local_scalar_t__> sigma_N_flat__;
        current_statement__ = 25;
        sigma_N_flat__ = context__.vals_r("sigma_N");
        current_statement__ = 25;
        pos__ = 1;
        current_statement__ = 25;
        for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
          current_statement__ = 25;
          assign(sigma_N, sigma_N_flat__[(pos__ - 1)],
            "assigning variable sigma_N", index_uni(sym1__));
          current_statement__ = 25;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 25;
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        current_statement__ = 25;
        check_greater_or_equal(function__, "sigma_N[sym1__]",
                               sigma_N[(sym1__ - 1)], 0);
      }
      current_statement__ = 26;
      validate_non_negative_index("sigma_C", "M", M);
      current_statement__ = 27;
      context__.validate_dims("data initialization","sigma_C","double",
           std::vector<size_t>{static_cast<size_t>(M)});
      sigma_C__ = Eigen::Matrix<double, -1, 1>(M);
      new (&sigma_C) Eigen::Map<Eigen::Matrix<double, -1, 1>>(sigma_C__.data(), M);
      
      
      {
        std::vector<local_scalar_t__> sigma_C_flat__;
        current_statement__ = 27;
        sigma_C_flat__ = context__.vals_r("sigma_C");
        current_statement__ = 27;
        pos__ = 1;
        current_statement__ = 27;
        for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
          current_statement__ = 27;
          assign(sigma_C, sigma_C_flat__[(pos__ - 1)],
            "assigning variable sigma_C", index_uni(sym1__));
          current_statement__ = 27;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 27;
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        current_statement__ = 27;
        check_greater_or_equal(function__, "sigma_C[sym1__]",
                               sigma_C[(sym1__ - 1)], 0);
      }
      current_statement__ = 28;
      validate_non_negative_index("beta", "D", D);
      current_statement__ = 29;
      validate_non_negative_index("gamma", "D", D);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    num_params_r__ = ((2 * (2 - 1)) / 2) + 2 + 2 + (D * 2) + D;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "linear_hierarchical_model_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      Eigen::Matrix<local_scalar_t__, -1, -1> Omega;
      Omega = Eigen::Matrix<local_scalar_t__, -1, -1>(2, 2);
      stan::math::fill(Omega, DUMMY_VAR__);
      
      current_statement__ = 1;
      Omega = in__.template read_constrain_corr_matrix<Eigen::Matrix<local_scalar_t__, -1, -1>, jacobian__>(
                lp__, 2);
      Eigen::Matrix<local_scalar_t__, -1, 1> sigma_beta;
      sigma_beta = Eigen::Matrix<local_scalar_t__, -1, 1>(2);
      stan::math::fill(sigma_beta, DUMMY_VAR__);
      
      current_statement__ = 2;
      sigma_beta = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                     0, lp__, 2);
      Eigen::Matrix<local_scalar_t__, -1, 1> mu;
      mu = Eigen::Matrix<local_scalar_t__, -1, 1>(2);
      stan::math::fill(mu, DUMMY_VAR__);
      
      current_statement__ = 3;
      mu = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
             0, lp__, 2);
      std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>> beta;
      beta = std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>(D, Eigen::Matrix<local_scalar_t__, -1, 1>(2));
      stan::math::fill(beta, DUMMY_VAR__);
      
      current_statement__ = 4;
      beta = in__.template read_constrain_lb<std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>, jacobian__>(
               0, lp__, D, 2);
      Eigen::Matrix<local_scalar_t__, -1, 1> gamma;
      gamma = Eigen::Matrix<local_scalar_t__, -1, 1>(D);
      stan::math::fill(gamma, DUMMY_VAR__);
      
      current_statement__ = 5;
      gamma = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                0, lp__, D);
      {
        current_statement__ = 6;
        lp_accum__.add(lkj_corr_lpdf<propto__>(Omega, 2));
        current_statement__ = 7;
        lp_accum__.add(exponential_lpdf<propto__>(sigma_beta, 1));
        current_statement__ = 8;
        lp_accum__.add(
          multi_normal_lpdf<propto__>(beta, mu,
            quad_form_diag(Omega, sigma_beta)));
        local_scalar_t__ CH4_hat;
        CH4_hat = DUMMY_VAR__;
        
        local_scalar_t__ Q;
        Q = DUMMY_VAR__;
        
        current_statement__ = 15;
        for (int i = 1; i <= M; ++i) {
          current_statement__ = 11;
          CH4_hat = (rvalue(beta, "beta",
                       index_uni(rvalue(day_id, "day_id", index_uni(i))),
                         index_uni(1)) +
                      (rvalue(beta, "beta",
                         index_uni(rvalue(day_id, "day_id", index_uni(i))),
                           index_uni(2)) *
                        rvalue(NO2_obs, "NO2_obs", index_uni(i))));
          current_statement__ = 12;
          Q = stan::math::sqrt(
                ((square(
                    (rvalue(beta, "beta",
                       index_uni(rvalue(day_id, "day_id", index_uni(i))),
                         index_uni(2)) *
                      rvalue(sigma_N, "sigma_N",
                        index_uni(rvalue(day_id, "day_id", index_uni(i))))))
                   +
                   square(
                     rvalue(gamma, "gamma",
                       index_uni(rvalue(day_id, "day_id", index_uni(i)))))) +
                  square(
                    rvalue(sigma_C, "sigma_C",
                      index_uni(rvalue(day_id, "day_id", index_uni(i)))))));
          current_statement__ = 13;
          lp_accum__.add(
            normal_lpdf<propto__>(rvalue(CH4_obs, "CH4_obs", index_uni(i)),
              CH4_hat, Q));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_std_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.resize(0);
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "linear_hierarchical_model_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      Eigen::Matrix<double, -1, -1> Omega;
      Omega = Eigen::Matrix<double, -1, -1>(2, 2);
      stan::math::fill(Omega, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 1;
      Omega = in__.template read_constrain_corr_matrix<Eigen::Matrix<local_scalar_t__, -1, -1>, jacobian__>(
                lp__, 2);
      Eigen::Matrix<double, -1, 1> sigma_beta;
      sigma_beta = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(sigma_beta, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 2;
      sigma_beta = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                     0, lp__, 2);
      Eigen::Matrix<double, -1, 1> mu;
      mu = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(mu, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 3;
      mu = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
             0, lp__, 2);
      std::vector<Eigen::Matrix<double, -1, 1>> beta;
      beta = std::vector<Eigen::Matrix<double, -1, 1>>(D, Eigen::Matrix<double, -1, 1>(2));
      stan::math::fill(beta, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 4;
      beta = in__.template read_constrain_lb<std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>, jacobian__>(
               0, lp__, D, 2);
      Eigen::Matrix<double, -1, 1> gamma;
      gamma = Eigen::Matrix<double, -1, 1>(D);
      stan::math::fill(gamma, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 5;
      gamma = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                0, lp__, D);
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        for (int sym2__ = 1; sym2__ <= 2; ++sym2__) {
          vars__.emplace_back(
            rvalue(Omega, "Omega", index_uni(sym2__), index_uni(sym1__)));
        }
      }
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        vars__.emplace_back(sigma_beta[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        vars__.emplace_back(mu[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        for (int sym2__ = 1; sym2__ <= D; ++sym2__) {
          vars__.emplace_back(beta[(sym2__ - 1)][(sym1__ - 1)]);
        }
      }
      for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
        vars__.emplace_back(gamma[(sym1__ - 1)]);
      }
      if (logical_negation((primitive_value(emit_transformed_parameters__) ||
            primitive_value(emit_generated_quantities__)))) {
        return ;
      } 
      if (logical_negation(emit_generated_quantities__)) {
        return ;
      } 
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_std_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(const stan::io::var_context& context__,
                                   VecI& params_i__, VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.clear();
    vars__.reserve(num_params_r__);
    int current_statement__ = 0; 
    
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      Eigen::Matrix<double, -1, -1> Omega;
      Omega = Eigen::Matrix<double, -1, -1>(2, 2);
      stan::math::fill(Omega, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> Omega_flat__;
        current_statement__ = 1;
        Omega_flat__ = context__.vals_r("Omega");
        current_statement__ = 1;
        pos__ = 1;
        current_statement__ = 1;
        for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
          current_statement__ = 1;
          for (int sym2__ = 1; sym2__ <= 2; ++sym2__) {
            current_statement__ = 1;
            assign(Omega, Omega_flat__[(pos__ - 1)],
              "assigning variable Omega", index_uni(sym2__),
                                            index_uni(sym1__));
            current_statement__ = 1;
            pos__ = (pos__ + 1);
          }
        }
      }
      Eigen::Matrix<double, -1, 1> Omega_free__;
      Omega_free__ = Eigen::Matrix<double, -1, 1>(((2 * (2 - 1)) / 2));
      stan::math::fill(Omega_free__, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 1;
      assign(Omega_free__, stan::math::corr_matrix_free(Omega),
        "assigning variable Omega_free__");
      Eigen::Matrix<double, -1, 1> sigma_beta;
      sigma_beta = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(sigma_beta, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> sigma_beta_flat__;
        current_statement__ = 2;
        sigma_beta_flat__ = context__.vals_r("sigma_beta");
        current_statement__ = 2;
        pos__ = 1;
        current_statement__ = 2;
        for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
          current_statement__ = 2;
          assign(sigma_beta, sigma_beta_flat__[(pos__ - 1)],
            "assigning variable sigma_beta", index_uni(sym1__));
          current_statement__ = 2;
          pos__ = (pos__ + 1);
        }
      }
      Eigen::Matrix<double, -1, 1> sigma_beta_free__;
      sigma_beta_free__ = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(sigma_beta_free__, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 2;
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        current_statement__ = 2;
        assign(sigma_beta_free__,
          stan::math::lb_free(sigma_beta[(sym1__ - 1)], 0),
          "assigning variable sigma_beta_free__", index_uni(sym1__));
      }
      Eigen::Matrix<double, -1, 1> mu;
      mu = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(mu, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> mu_flat__;
        current_statement__ = 3;
        mu_flat__ = context__.vals_r("mu");
        current_statement__ = 3;
        pos__ = 1;
        current_statement__ = 3;
        for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
          current_statement__ = 3;
          assign(mu, mu_flat__[(pos__ - 1)],
            "assigning variable mu", index_uni(sym1__));
          current_statement__ = 3;
          pos__ = (pos__ + 1);
        }
      }
      Eigen::Matrix<double, -1, 1> mu_free__;
      mu_free__ = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(mu_free__, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 3;
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        current_statement__ = 3;
        assign(mu_free__, stan::math::lb_free(mu[(sym1__ - 1)], 0),
          "assigning variable mu_free__", index_uni(sym1__));
      }
      std::vector<Eigen::Matrix<double, -1, 1>> beta;
      beta = std::vector<Eigen::Matrix<double, -1, 1>>(D, Eigen::Matrix<double, -1, 1>(2));
      stan::math::fill(beta, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> beta_flat__;
        current_statement__ = 4;
        beta_flat__ = context__.vals_r("beta");
        current_statement__ = 4;
        pos__ = 1;
        current_statement__ = 4;
        for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
          current_statement__ = 4;
          for (int sym2__ = 1; sym2__ <= D; ++sym2__) {
            current_statement__ = 4;
            assign(beta, beta_flat__[(pos__ - 1)],
              "assigning variable beta", index_uni(sym2__), index_uni(sym1__));
            current_statement__ = 4;
            pos__ = (pos__ + 1);
          }
        }
      }
      std::vector<Eigen::Matrix<double, -1, 1>> beta_free__;
      beta_free__ = std::vector<Eigen::Matrix<double, -1, 1>>(D, Eigen::Matrix<double, -1, 1>(2));
      stan::math::fill(beta_free__, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 4;
      for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
        current_statement__ = 4;
        for (int sym2__ = 1; sym2__ <= 2; ++sym2__) {
          current_statement__ = 4;
          assign(beta_free__,
            stan::math::lb_free(beta[(sym1__ - 1)][(sym2__ - 1)], 0),
            "assigning variable beta_free__", index_uni(sym1__),
                                                index_uni(sym2__));
        }
      }
      Eigen::Matrix<double, -1, 1> gamma;
      gamma = Eigen::Matrix<double, -1, 1>(D);
      stan::math::fill(gamma, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> gamma_flat__;
        current_statement__ = 5;
        gamma_flat__ = context__.vals_r("gamma");
        current_statement__ = 5;
        pos__ = 1;
        current_statement__ = 5;
        for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
          current_statement__ = 5;
          assign(gamma, gamma_flat__[(pos__ - 1)],
            "assigning variable gamma", index_uni(sym1__));
          current_statement__ = 5;
          pos__ = (pos__ + 1);
        }
      }
      Eigen::Matrix<double, -1, 1> gamma_free__;
      gamma_free__ = Eigen::Matrix<double, -1, 1>(D);
      stan::math::fill(gamma_free__, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 5;
      for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
        current_statement__ = 5;
        assign(gamma_free__, stan::math::lb_free(gamma[(sym1__ - 1)], 0),
          "assigning variable gamma_free__", index_uni(sym1__));
      }
      for (int sym1__ = 1; sym1__ <= ((2 * (2 - 1)) / 2); ++sym1__) {
        vars__.emplace_back(Omega_free__[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        vars__.emplace_back(sigma_beta_free__[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        vars__.emplace_back(mu_free__[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
        for (int sym2__ = 1; sym2__ <= 2; ++sym2__) {
          vars__.emplace_back(beta_free__[(sym1__ - 1)][(sym2__ - 1)]);
        }
      }
      for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
        vars__.emplace_back(gamma_free__[(sym1__ - 1)]);
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"Omega", "sigma_beta", "mu", "beta",
      "gamma"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{
                                                                   static_cast<size_t>(2)
                                                                   ,
                                                                   static_cast<size_t>(2)
                                                                   },
      std::vector<size_t>{static_cast<size_t>(2)},
      std::vector<size_t>{static_cast<size_t>(2)},
      std::vector<size_t>{static_cast<size_t>(D), static_cast<size_t>(2)},
      std::vector<size_t>{static_cast<size_t>(D)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        for (int sym2__ = 1; sym2__ <= 2; ++sym2__) {
          {
            param_names__.emplace_back(std::string() + "Omega" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
          } 
        }
      } 
    }
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "sigma_beta" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "mu" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        for (int sym2__ = 1; sym2__ <= D; ++sym2__) {
          {
            param_names__.emplace_back(std::string() + "beta" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
          } 
        }
      } 
    }
    for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "gamma" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= ((2 * (2 - 1)) / 2); ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "Omega" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "sigma_beta" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "mu" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        for (int sym2__ = 1; sym2__ <= D; ++sym2__) {
          {
            param_names__.emplace_back(std::string() + "beta" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
          } 
        }
      } 
    }
    for (int sym1__ = 1; sym1__ <= D; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "gamma" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"Omega\",\"type\":{\"name\":\"matrix\",\"rows\":" + std::to_string(2) + ",\"cols\":" + std::to_string(2) + "},\"block\":\"parameters\"},{\"name\":\"sigma_beta\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"parameters\"},{\"name\":\"mu\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(D) + ",\"element_type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "}},\"block\":\"parameters\"},{\"name\":\"gamma\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(D) + "},\"block\":\"parameters\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"Omega\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(((2 * (2 - 1)) / 2)) + "},\"block\":\"parameters\"},{\"name\":\"sigma_beta\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"parameters\"},{\"name\":\"mu\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(D) + ",\"element_type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "}},\"block\":\"parameters\"},{\"name\":\"gamma\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(D) + "},\"block\":\"parameters\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      std::vector<double> vars_vec;
      vars_vec.reserve(vars.size());
      std::vector<int> params_i;
      write_array_impl(base_rng, params_r, params_i, vars_vec,
          emit_transformed_parameters, emit_generated_quantities, pstream);
      vars = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        vars_vec.data(), vars_vec.size());
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      write_array_impl(base_rng, params_r, params_i, vars,
       emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec;
      params_r_vec.reserve(params_r.size());
      std::vector<int> params_i;
      transform_inits_impl(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }
    inline void transform_inits(const stan::io::var_context& context,
                                std::vector<int>& params_i,
                                std::vector<double>& vars,
                                std::ostream* pstream = nullptr) const final {
      transform_inits_impl(context, params_i, vars, pstream);
    }

};
}

using stan_model = linear_hierarchical_model_model_namespace::linear_hierarchical_model_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return linear_hierarchical_model_model_namespace::profiles__;
}

#endif



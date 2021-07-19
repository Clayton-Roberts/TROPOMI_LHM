data {
  int<lower=1>          M;
  vector[M]             CH4_obs;
  vector[M]             NO2_obs;
  vector<lower=0>[M]    sigma_N;   // Observational error on NO2.
  vector<lower=0>[M]    sigma_C;   // Observational error on CH4.
  int<lower=1>          D;
  int<lower=1, upper=D> day_id[M];
}
parameters {
  vector<lower=0>[D]      gamma;    // Group d variance term.
  vector<lower=0>[2]      tau_u;    // Standard deviations in the covariance matrix.
  real                    mu_alpha;
  real                    mu_beta;
  matrix[2, D]            z_u;
  cholesky_factor_corr[2] L_u;
}
transformed parameters {
  matrix[D, 2] u;
  u = (diag_pre_multiply(tau_u, L_u) * z_u)';
}
model {
    // Priors
    target += normal_lpdf(mu_alpha| 1865,10);
    target += normal_lpdf(mu_beta | 1.0,0.3);
    // For now, no prior on group d variance term gamma.
    //target += normal_lpdf(sigma | 0, 5)  -
        //normal_lccdf(0 | 0, 5);
    target += normal_lpdf(tau_u[1] | 0, 20)  -
        normal_lccdf(0 | 0, 20);
    target += normal_lpdf(tau_u[2] | 0, 0.5)  -
        normal_lccdf(0 | 0, 0.5);
    target += lkj_corr_cholesky_lpdf(L_u | 2);
    target += std_normal_lpdf(to_vector(z_u));

    // Likelihood
    target += normal_lpdf(CH4_obs | mu_alpha + u[day_id, 1] +
                        NO2_obs .* (mu_beta + u[day_id, 2]),
                        sqrt(square(gamma[day_id]) +
                        square(sigma_C) +
                        square(sigma_N .* (mu_beta + u[day_id, 2]))));
}
generated quantities {
  corr_matrix[2] rho_u       = L_u * L_u';
  vector[D]      alpha       = u[,1];
  vector[D]      beta        = u[,2];
  real           sigma_alpha = tau_u[1];
  real           sigma_beta  = tau_u[2];
}
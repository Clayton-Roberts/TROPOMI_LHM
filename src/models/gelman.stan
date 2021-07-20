data {
  int<lower=0> M;                  // num individuals
  int<lower=1> K;                  // num ind predictors
  int<lower=1> D;                  // num groups
  int<lower=1,upper=D> day_id[M];  // group for individual
  matrix[M, K] NO2_obs;                  // individual predictors
  vector[M] CH4_obs;                     // outcomes
  vector<lower=0>[M]    sigma_N;	// Observational error on NO2.
  vector<lower=0>[M]    sigma_C;	// Observational error on CH4.
}
parameters {
  corr_matrix[K] Omega;        // prior correlation
  vector<lower=0>[K] tau;      // prior scale
  vector[K] kappa[D];           // indiv coeffs by group
  vector<lower=0>[D] gamma;    // group variance term
  vector[K] mu;                // mean vector
}
model {
  tau ~ cauchy(0, 2.5);
  Omega ~ lkj_corr(2);
  kappa ~ multi_normal(mu, quad_form_diag(Omega, tau));

  {
    vector[M] x_kappa_day_id;
    //vector[M] Q;
    for (i in 1:M)
      x_kappa_day_id[i] = NO2_obs[i] * kappa[day_id[i]];
      //Q[i] = sqrt(square(kappa[day_id[i], 2] * sigma_N[i]) + square(gamma[day_id[i]]) + square(sigma_C[i]));

    CH4_obs ~ normal(x_kappa_day_id,
                     sqrt(square(gamma[day_id]) +
                        square(sigma_C) +
                        square(to_vector(kappa[day_id, 2]) .* sigma_N )));
  }
}
generated quantities {
    vector[D]    alpha;
    vector[D]    beta;
    real         mu_alpha;
    real         mu_beta;
    real         rho;
    real         sigma_alpha;
    real         sigma_beta;

    for (d in 1:D) {
        alpha[d] = kappa[d, 1];
        beta[d]  = kappa[d, 2];
    }

    mu_alpha = mu[1];
    mu_beta  = mu[2];

    rho = Omega[1, 2];

    sigma_alpha = tau[1];
    sigma_beta  = tau[2];
}
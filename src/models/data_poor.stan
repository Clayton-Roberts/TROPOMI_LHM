// Note: In the paper and code, our daily intercept is alpha and daily slope is beta.
// Here, kappa is a vector and kappa[d, 1] is the day d intercept and kappa[d, 2] is the day d slope.
// It's just easier to have it in vector form. Generated quantities block produces alpha and beta from kappa for consistency.
data {
    int<lower=0>            N;                // Total number of observations, i.e, rows in the dataset csv file.
    vector[N]               NO2_obs;	      // Observations of NO2.
    vector[N]               CH4_obs;          // Observations of CH4.
    real<lower=0>           sigma_N;	      // Observational error on NO2 (averaged over the day).
    real<lower=0>           sigma_C;	      // Observational error on CH4 (averaged over the day).

    vector[5]               theta;            // Mean vector of postier estimates of mu_alpha, mu_beta, Sigma_1_1,
                                              // Sigma_1_2, and Sigma_2_2.
    cov_matrix[5]           Upsilon;          // Covariance matrix of the five elements stated above.
}
transformed data {
    matrix[5, 5] L;
    L = cholesky_decompose(Upsilon);
}
parameters {
    real<lower=0>           gamma;            // Daily variance term.
    vector[5]               epsilon;
    vector[2]               kappa;
}
transformed parameters {
    vector[5]      tau;
    matrix[2, 2]   Sigma;            // Covariance matrix of alpha and beta.
    vector[2]      mu;               // Mean vector of alpha and beta

    tau = theta + L * epsilon;
    mu[1] = tau[1];
    mu[2] = tau[2];
    Sigma[1, 1] = tau[3];
    Sigma[1, 2] = tau[4];
    Sigma[2, 1] = tau[4];
    Sigma[2, 2] = tau[5];
}
model{

    // implies: tau ~ multi_normal(theta, Upsilon)
    epsilon ~ std_normal();

    kappa ~ multi_normal(mu, Sigma);

    vector[N] CH4_hat = kappa[1] + (kappa[2] * NO2_obs);
    real      sigma   = sqrt(square(gamma) + square(sigma_C) + square(kappa[2] * sigma_N));

    CH4_obs ~ normal(CH4_hat, sigma);
}
generated quantities {
    real alpha;
    real beta;

    alpha = kappa[1];
    beta  = kappa[2];
}
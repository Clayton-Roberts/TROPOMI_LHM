// Note: In the paper and code, our daily intercept is alpha and daily slope is beta.
// Here, kappa is a vector and kappa[d, 1] is the day d intercept and kappa[d, 2] is the day d slope.
// It's just easier to have it in vector form. Generated quantities block produces alpha and beta from kappa for consistency.
data {
    int<lower=0>          M;         // Total number of observations, i.e, rows in the dataset csv file.
    int<lower=0>          D;         // Number of days we have observations on.
    int<lower=1,upper=D>  day_id[M]; // Vector of day ids, which will be an integer between 1 and D inclusive.
    vector[M]             NO2_obs;   // Observations of NO2.
    vector[M]             CH4_obs;   // Observations of CH4.
    vector<lower=0>[M]    sigma_N;   // Observational error on NO2.
    vector<lower=0>[M]    sigma_C;   // Observational error on CH4.
}
parameters {
    corr_matrix[2]     Omega;       // Correlation matrix.
    vector<lower=0>[2] sigma_kappa;  // Standard deviations in the covariance matrix.
    vector<lower=0>[2] mu;          // Intercept (mu[1]) and slope (mu[2]) hyperparameters.
    vector<lower=0>[2] kappa[D];     // Group d intercept (kappa[d, 1]) and group d slope (kappa[d, 2]).
    vector<lower=0>[D] gamma;       // Group d variance term.
}
model {

    // Priors for our hyper-parameters, mu has a flat prior.
    Omega      ~ lkj_corr(2);
    sigma_kappa ~ exponential(1);

    // Model assumption is that beta[1] and beta[2] are drawn from a multivariate normal distribution.
    kappa ~ multi_normal(mu, quad_form_diag(Omega, sigma_kappa));

    real CH4_hat;
    real Q;

    for (i in 1:M) {
        CH4_hat    = kappa[day_id[i], 1] + kappa[day_id[i], 2] * NO2_obs[i];
        Q          = sqrt(square(kappa[day_id[i], 2] * sigma_N[day_id[i]]) + square(gamma[day_id[i]]) + square(sigma_C[day_id[i]]));
        CH4_obs[i] ~ normal(CH4_hat, Q);
    }
}
generated quantities {
    matrix[2, 2] Sigma;
    vector[D]    alpha;
    vector[D]    beta;
    real         mu_alpha;
    real         mu_beta;

    Sigma[1, 1] = square(sigma_kappa[1]);
    Sigma[1, 2] = sigma_kappa[1] * sigma_kappa[2] * Omega[1, 2];
    Sigma[2, 1] = sigma_kappa[2] * sigma_kappa[1] * Omega[2, 1];
    Sigma[2, 2] = square(sigma_kappa[2]);

    for (d in 1:D) {
        alpha[d] = kappa[d, 1];
        beta[d]  = kappa[d, 2];
    }

    mu_alpha = mu[1];
    mu_beta  = mu[2];
}
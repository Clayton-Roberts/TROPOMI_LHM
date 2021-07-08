// Note: In the paper and code, our daily intercept is alpha and daily slope is beta.
// Here, beta is a vector and beta[d, 1] is the day d intercept and beta[d, 2] is the day d slope.
// It's just easier to have it in vector form.
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
    vector<lower=0>[2] sigma_beta;  // Standard deviations in the covariance matrix.
    vector<lower=0>[2] mu;          // Intercept (mu[1]) and slope (mu[2]) hyperparameters.
    vector<lower=0>[2] beta[D];     // Group d intercept (beta[d, 1]) and group d slope (beta[d, 2]).
    vector<lower=0>[D] gamma;       // Group d variance term.
}
model {

    // Priors for our hyper-parameters, mu has a flat prior.
    Omega      ~ lkj_corr(2);
    sigma_beta ~ exponential(1);

    // Model assumption is that beta[1] and beta[2] are drawn from a multivariate normal distribution.
    beta ~ multi_normal(mu, quad_form_diag(Omega, sigma_beta));

    real CH4_hat;
    real Q;

    for (i in 1:M) {
        CH4_hat    = beta[day_id[i], 1] + beta[day_id[i], 2] * NO2_obs[i];
        Q          = sqrt(square(beta[day_id[i], 2] * sigma_N[day_id[i]]) + square(gamma[day_id[i]]) + square(sigma_C[day_id[i]]));
        CH4_obs[i] ~ normal(CH4_hat, Q);
    }
}
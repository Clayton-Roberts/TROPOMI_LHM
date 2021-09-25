// Note: In the paper and code, our daily intercept is alpha and daily slope is beta.
// Here, kappa is a vector and kappa[d, 1] is the day d intercept and kappa[d, 2] is the day d slope.
// It's just easier to have it in vector form. Generated quantities block produces alpha and beta from kappa for consistency.
data {
    int<lower=0>            N;                // Total number of observations, i.e, rows in the dataset csv file.
    int<lower=0>            D;                // Number of days we have observations on.
    int<lower=1>            group_sizes[D];   // Number of observations in each day.
    int<lower=1,upper=D>    day_id[N];        // Vector of day ids, which will be an integer between 1 and D inclusive.
    vector[N]               NO2_obs;	      // Observations of NO2.
    vector[N]               CH4_obs;          // Observations of CH4.
    vector<lower=0>[N]      sigma_N;	      // Observational error on NO2 (per individual observation).
    vector<lower=0>[N]      sigma_C;	      // Observational error on CH4 (per individual observation).
}
parameters {
    vector[2]               epsilon[D];
    corr_matrix[2]          Omega;            // Correlation matrix.
    vector<lower=0>[2]      sigma_kappa;      // Standard deviations in the covariance matrix.
    vector<lower=0>[2]      mu;               // Intercept (mu[1]) and slope (mu[2]) hyperparameters.
    vector<lower=0>[D]      gamma;            // Group d variance term.
}
transformed parameters {
    matrix[2, 2]            L;                // Cholesky decomposition of the correlation matrix.
    vector<lower=0>[2]      kappa[D];         // Group d intercept (kappa[d, 1]) and group d slope (kappa[d, 2]).

    L = cholesky_decompose(Omega);

    for (d in 1:D)
        kappa[d] = mu + sigma_kappa .* (L * epsilon[d]);
}
model{
    int pos;
    pos = 1;

    // Priors for our hyper-parameters. All others have flat priors.
    sigma_kappa ~ exponential(1);

    // The following line implies kappa ~ multi_normal(mu, quad_form_diag(Omega, sigma_kappa))
    for (d in 1:D) epsilon[d] ~ std_normal();

    vector[N] CH4_hat = to_vector(kappa[day_id, 1]) + (to_vector(kappa[day_id, 2]) .* NO2_obs);
    vector[N] sigma   = sqrt(square(gamma[day_id]) + square(sigma_C) + square(to_vector(kappa[day_id, 2]) .* sigma_N));

    CH4_obs ~ normal(CH4_hat, sigma);
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

    sigma_alpha = sigma_kappa[1];
    sigma_beta  = sigma_kappa[2];
}

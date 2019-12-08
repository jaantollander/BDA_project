data {
  int<lower=0> N; // number of data points
  int<lower=0> K; // number of groups
  int<lower=1,upper=K> x[N]; // group indicator
  vector[N] sales; // sales
  vector[N] scores; // critic scores
  real pmu_scores; // prior mean
  real psigma_scores; // prior std
}
parameters {
  vector[K] mu_sales;
  vector[K] mu_scores;
  real<lower=0> sigma_sales[K];
  real<lower=0> sigma_scores[K];
}
model {
  mu_scores ~ normal(pmu_scores, psigma_scores);
  sales ~ normal(mu_sales[x], sigma_sales[x]);
  scores ~ normal(mu_scores[x], sigma_scores[x]);
}
generated quantities {
  vector[N] sales_log_lik;
  vector[N] scores_log_lik;
  vector[K] sales_pred;
  vector[K] scores_pred;
  for (i in 1:N) {
    sales_log_lik[i] = normal_lpdf(sales[i] | mu_sales[x[i]], sigma_sales[x[i]]);
  }
  for (i in 1:N) {
    scores_log_lik[i] = normal_lpdf(scores[i] | mu_scores[x[i]], sigma_scores[x[i]]);
  }
  for (i in 1:K) {
    sales_pred[i] = normal_rng(mu_sales[i], sigma_sales[i]);
  }
  for (i in 1:K) {
    scores_pred[i] = normal_rng(mu_scores[i], sigma_scores[i]);
  }
}

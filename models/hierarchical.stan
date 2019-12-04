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
  real<lower=0> sigma_sales;
  real<lower=0> sigma_scores;
}
model {
  mu_scores ~ normal(pmu_scores, psigma_scores);
  sales ~ normal(mu_sales[x], sigma_sales);
  scores ~ normal(mu_scores[x], sigma_scores);
}
generated quantities {
  vector[N] sales_log_lik;
  vector[N] scores_log_lik;
  for (i in 1:N) {
    sales_log_lik[i] = normal_lpdf(sales[i] | mu_sales[x[i]], sigma_sales);
  }
  for (i in 1:N) {
    scores_log_lik[i] = normal_lpdf(scores[i] | mu_scores[x[i]], sigma_scores);
  }
}

library(fastRG)
library(EnvStats)
set.seed(123)

n = 2000
k = 10
within <- 0.28
between <- 0.08
B <- matrix(between, k, k)
diag(B) <- within
n_samples <- 10

exp_degrees <- seq(from=25, to=60, by=5)

for(degree in exp_degrees){
  for(i in 1:n_samples){
    theta_point <- rep(1/n, n)
    theta_exp <- rexp(n, 5)
    theta_exp <- theta_exp / sum(theta_exp)
    theta_pareto <- rpareto(n, 0.5, 5)
    theta_pareto <- theta_pareto / sum(theta_pareto)
    
    sbm_point <- dcsbm(theta = theta_point, B = B, expected_degree = degree)
    A <- as.matrix(sample_sparse(sbm_point, poisson_edges = F, allow_self_loops = F))
    write.table(A, paste0('samples/graph_point_', degree, '_', i, '.table'))
    
    sbm_exp <- dcsbm(theta = theta_exp, B = B, expected_degree = degree)
    A <- as.matrix(sample_sparse(sbm_exp, poisson_edges = F, allow_self_loops = F))
    write.table(A, paste0('samples/graph_exp_', degree, '_', i, '.table'))
    
    sbm_pareto <- dcsbm(theta = theta_pareto, B = B, expected_degree = degree)
    A <- as.matrix(sample_sparse(sbm_pareto, poisson_edges = F, allow_self_loops = F))
    write.table(A, paste0('samples/graph_pareto_', degree, '_', i, '.table'))
    
  }
}

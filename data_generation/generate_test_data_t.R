library(fastRG)
library(EnvStats)
library(Matrix)
set.seed(123)

n = 2000
k = 2
within <- 0.25
between <- 0.1
B <- matrix(between, k, k)
diag(B) <- within
n_samples <- 100

exp_degrees <- seq(from=3.5, to=105, by=3.5)

for(degree in exp_degrees){
  for(i in 1:n_samples){
    theta_point <- rep(1/n, n)

    
    sbm_point <- dcsbm(theta = theta_point, B = B, expected_degree = degree)
    A <- as.matrix(sample_sparse(sbm_point, poisson_edges = F, allow_self_loops = F))
    # write.table(A, paste0('samples_t/graph_Bernoulli_', degree, '_', i, '.table'))
    writeMM(sample_sparse(sbm_point, poisson_edges = F, allow_self_loops = F), paste0('samples_t/graph_Bernoulli_', degree, '_', i, '.table'))
    
    A <- as.matrix(sample_sparse(sbm_point, poisson_edges = T, allow_self_loops = F))
    # write.table(A, paste0('samples_t/graph_Poisson_', degree, '_', i, '.table'))
    writeMM(sample_sparse(sbm_point, poisson_edges = T, allow_self_loops = F), paste0('samples_t/graph_Poisson_', degree, '_', i, '.table'))
    print(c(i, degree))
  }
}

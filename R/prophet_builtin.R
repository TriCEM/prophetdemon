## .................................................................................
## Purpose: Using tensor_flow and reticulate to build the prophet_bi
##
## Author: Nick Brazeau & David Rasmussen
##
## Date: 20 February, 2023
##
## Notes:
## reticulate and TF for R: https://rstudio-pubs-static.s3.amazonaws.com/529704_0a08ca3509cd4990bb44014fbea096ad.html#18
## TF Repo for R: https://tensorflow.rstudio.com/install/
## https://tensorflow.rstudio.com/guides/keras/writing_a_training_loop_from_scratch
## .................................................................................
#install.packages("tensorflow")
#install.packages("keras")
library(tidyverse)
library(reticulate)
library(tensorflow)
library(keras)
reticulate::use_python("/Users/nbrazeau/Documents/Github/prophetdemon/venv/bin/python")
reticulate::use_virtualenv("/Users/nbrazeau/Documents/Github/prophetdemon/venv")

#............................................................
# Read in Simulation Data
#...........................................................
#............................................................
# Make Some Simulation
#...........................................................
# create training data randomly
n <- 100
train_size <- 1e2
rewireprob <- rnbinom(n = train_size, size = 5, mu = 25)/n
conmats <- lapply(rewireprob, function(x, n){igraph::as_adjacency_matrix(
  igraph::erdos.renyi.game(n = n,
                           p.or.m = x,
                           type = "gnp"), sparse = F)}, n = n)
run_fomes <- function(conmat, n){
  ret <- fomes::sim_Gillespie_SIR(Iseed = 1,
                                  N = n,
                                  beta = rep(0.4, n),
                                  dur_I = 8,
                                  rho = 1e-27,
                                  init_contact_mat = conmat,
                                  term_time = 500)
  return( sum(ret$Event_traj == "transmission")/n ) }

#......................
# run out
#......................
reps <- 1:100
train_dat <- tibble::tibble(rewireprob = rewireprob, n = n)
train_dat$conmat <- lapply(conmats, function(x){return(x)})
train_dat <- tidyr::expand_grid(train_dat, reps)
train_dat$finalsize <- purrr::pmap_dbl(train_dat[,c("conmat", "n")],
                                       run_fomes)
# pull out columns
train_dat_net <- tensorflow::as_tensor( train_dat$conmat )
true_final_sizes <- tensorflow::as_tensor( train_dat$finalsize )
true_final_sizes <- tensorflow::array_reshape(true_final_sizes, dim = c(nrow(train_dat), 1))


#............................................................
# Set up prophet_bi model
#   For Conv layers: first argument is the # of filters, second argument is the kernel size
#   To think about: Should we sort rows/columns in adjacency matrix so most well-connected nodes are closer
#   Maybe by single-linkage clustering: https://www.section.io/engineering-education/hierarchical-clustering-in-python/
#...........................................................
n <- 1e2 # population size, needs to be same from sim
batch_size <- 10
input_shape_dim <- c(batch_size, n, n, 1)
# make CNN model
prophet_bi <- keras::keras_model_sequential() %>%
  keras::layer_conv_2d(., filters = 2, kernel_size = 8, activation = "relu", padding = "same", input_shape = input_shape_dim[2:length(input_shape_dim)]) %>%
  keras::layer_max_pooling_2d(pool_size = 2) %>%
  keras::layer_conv_2d(., filters = 4, kernel_size = 4, activation = "relu", padding = "same") %>%
  keras::layer_max_pooling_2d(pool_size = 2) %>%
  keras::layer_conv_2d(., filters = 8, kernel_size = 2, activation = "relu", padding = "same") %>%
  keras::layer_max_pooling_2d(pool_size = 2) %>%
  keras::layer_flatten() %>%
  keras::layer_dense(., units = 32, activation = "relu") %>%
  keras::layer_dense(., units = 16, activation = "relu") %>%
  keras::layer_dense(., units = 1, activation = "sigmoid")

# see summary of what we made
summary(prophet_bi)





#............................................................
# Using built in
#...........................................................

#true_final_sizes <- tensorflow::array_reshape(true_final_sizes, dim = c(batch_size, 1))

prophet_bi %>% compile(
  optimizer = keras::optimizer_adam(),  # Optimizer
  # Loss function to minimize
  loss = keras::loss_mean_squared_error(),
  # List of metrics to monitor
  metrics = list(keras::metric_mean_absolute_error())
)

history <- prophet_bi %>%
  keras::fit(
    train_dat_net,
    true_final_sizes,
    batch_size = 10,
    epochs = 2
  )

history$metrics$loss
history$params

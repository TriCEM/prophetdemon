## .................................................................................
## Purpose: Using tensor_flow and reticulate to build the prophet_man_man
##
## Author: Nick Brazeau
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
library(tfautograph)
library(fomes)
reticulate::use_python("/Users/nbrazeau/Documents/Github/prophetdemon/venv/bin/python")
reticulate::use_virtualenv("/Users/nbrazeau/Documents/Github/prophetdemon/venv")

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
train_dat$conmat <- lapply(1:length(conmats), function(x){return(conmats[[x]])})
train_dat <- tidyr::expand_grid(train_dat, reps)
train_dat$finalsize <- purrr::pmap_dbl(train_dat[,c("conmat", "n")],
                                       run_fomes)
# pull out columns
train_dat_net <- train_dat$conmat
train_dat_fs <- train_dat$finalsize

#............................................................
# Set up prophet_man model
#   For Conv layers: first argument is the # of filters, second argument is the kernel size
#   To think about: Should we sort rows/columns in adjacency matrix so most well-connected nodes are closer
#   Maybe by single-linkage clustering: https://www.section.io/engineering-education/hierarchical-clustering-in-python/
#...........................................................
batch_size <- 10 # num sims per training episode
input_shape_dim <- c(batch_size, n, n, 1)
# make CNN model
prophet_man <- keras::keras_model_sequential() %>%
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
prophet_man


#............................................................
# Set up Training Regimen
#...........................................................
#......................
# gradient items
#......................
optimizer <- keras::optimizer_adam(learning_rate = 1e-3) # adam is good GD optimizer - can explore adamax, etc
loss_fn <- keras::loss_mean_squared_error() # can later change loss fxn

#......................
# setting up epoch/epochs
#......................
epochs <- 50 # number of times to explore the batched training data
epochs_losses <- rep(NA, epochs)
# slice the data into “batches” of size batch_size, and repeatedly iterating over the entire dataset for a given number of epochs.

# run through epochs
for (i in 1:epochs) {
  # Iterate over the batches of the dataset.
  batches <- sample(1:nrow(train_dat), size = nrow(train_dat), replace = F)
  batches <- split(batches, factor(sort(rep(1:(length(batches)/batch_size), batch_size))))
  # init batch loss
  batch_loss <- rep(NA, length(batches))
  for (j in 1:length(batches)) {
    net_batch <- tensorflow::as_tensor( train_dat_net[ batches[[j]] ] )
    true_final_sizes <- tensorflow::as_tensor( train_dat_fs[ batches[[j]] ] )
    true_final_sizes <- tensorflow::array_reshape(true_final_sizes, dim = c(batch_size, 1))

    # calculate gradients
    with(tf$GradientTape() %as% tape, {
      predicted_final_sizes <- prophet_man(net_batch)
      loss_value <- loss_fn(true_final_sizes, predicted_final_sizes)
    })
    # apply gradients
    grads <- tape$gradient(loss_value, prophet_man$trainable_variables)
    # zip together for gradients
    optimizer$apply_gradients(zip_lists(grads, prophet_man$trainable_variables))
    # store batch loss
    batch_loss[j] <- as.numeric(loss_value)
  }
  # store epoch loss
  epochs_losses[i] <- sum(batch_loss)
}

plot(epochs_losses)

#............................................................
# Validation Regimen
#...........................................................

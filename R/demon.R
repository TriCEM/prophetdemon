## .................................................................................
## Purpose: Demon variatonal autoencoder to thwart prophet
##
## Author: Nick Brazeau
##
## Date: 22 February, 2023
##
## Notes:
## VAE built heavily from https://github.com/ageron/handson-ml2
## tutorial on AE in R https://nbisweden.github.io/workshop-neural-nets-and-deep-learning/session_rAutoencoders/lab_autoencoder_hapmap.html
## https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
## .................................................................................
#install.packages("tensorflow")
#install.packages("keras")
library(tidyverse)
library(reticulate)
library(tensorflow)
library(keras)
reticulate::use_python("/Users/nbrazeau/Documents/Github/prophetdemon/venv/bin/python")
reticulate::use_virtualenv("/Users/nbrazeau/Documents/Github/prophetdemon/venv")


#++++++++++++++++++++++++++++++++++++++++++
###  functions     ####
#++++++++++++++++++++++++++++++++++++++++++
CodingSampler(keras::keras$layers$Layer) %py_class% {
  call <- function(inputs) {
    c(codings_mean, codings_log_var) %<-% inputs
    batch <- tf$shape(codings_mean)[1]
    dim <- tf$shape(codings_mean)[2]
    stdnorm <- keras::k_random_normal(shape = c(batch, dim))
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    # this log var is gamma pg 656
    codings_mean + exp(0.5 * codings_log_var) * stdnorm
  }
}



#++++++++++++++++++++++++++++++++++++++++++
### VAE Demon Model Setup        ####
#++++++++++++++++++++++++++++++++++++++++++
#............................................................
# Set up demon ENcoder
#...........................................................
Encoder(keras$layers$Layer) %py_class% {
  "Maps CNN to a triplet (z_mean, z_log_var, z)."

  initialize <- function(latent_dim = 32, intermediate_dim = 64,
                         name = "encoder", ...) {
    super$initialize(name = name, ...)
    self$codings_proj <- layer_dense(units = intermediate_dim,
                                     activation = "relu")
    self$codings_mean <- layer_dense(units = latent_dim)
    self$codings_log_var <- layer_dense(units = latent_dim)
    self$sampling <- CodingSampler()
  }

  call <- function(inputs) {
    x <- self$codings_proj(inputs)
    codings_mean <- self$codings_mean(x)
    codings_log_var <- self$codings_log_var(x)
    codings <- self$sampling(c(codings_mean, codings_log_var))
    list(codings_mean, codings_log_var, codings)
  }
}





#............................................................
# Set up demon DEcoder
#...........................................................
Decoder(keras$layers$Layer) %py_class% {
  "Converts z-codings, the encoded digit vector, back into a adjacency matrix."

  initialize <- function(original_dim,
                         intermediate_dim = 64,
                         name = "decoder", ...) {
    super$initialize(name = name, ...)
    self$codings_proj <- keras::layer_dense(units = intermediate_dim,
                                            activation = "relu")
    self$codings_output <- keras::layer_dense(units = original_dim,
                                              activation = "sigmoid")
  }

  call <- function(inputs) {
    x <- self$codings_proj(inputs)
    self$codings_output(x)
  }
}


#............................................................
# Set up demon combined EN-DEcoder VAE
#...........................................................
VariationalAutoEncoder(keras$Model) %py_class% {
  "Combines the encoder and decoder into an end-to-end VAE model for training."
  initialize <- function(original_dim, intermediate_dim = 64, latent_dim = 32,
                         name = "autoencoder", ...) {
    super$initialize(name = name, ...)
    self$original_dim <- original_dim

    self$encoder <- Encoder(
      latent_dim = latent_dim,
      intermediate_dim = intermediate_dim
    )
    self$decoder <- Decoder(
      original_dim,
      intermediate_dim = intermediate_dim
    )
  }

  call <- function(inputs) {
    c(codings_mean, codings_log_var, codings) %<-% self$encoder(inputs)
    reconstructed <- self$decoder(codings)
    # Add KL divergence regularization loss.
    kl_loss <- -0.5 * keras::k_sum(1 + codings_log_var - keras::k_exp(codings_log_var) - keras::k_square(codings_mean), axis=-1)
    self$add_loss(kl_loss)
    reconstructed
  }
}



#............................................................
# Set up VAE demon optimizer and learning rates
#...........................................................
N <- 10 # numn inidivid..uals
original_dim <- N^2
vae <- VariationalAutoEncoder(original_dim, 64, 32)


#++++++++++++++++++++++++++++++++++++++++++
### Training VAE Demon        ####
#++++++++++++++++++++++++++++++++++++++++++
#..............................
# generate training set
#...............................
train_size <- 1e1
rewireprob <- runif(train_size, min = 0.1, max = 0.5)
grphs <- sapply(rewireprob, function(x, n){
  out <- igraph::erdos.renyi.game(n = n,
                                  p.or.m = x,
                                  type = "gnp")
  out <- igraph::as_adjacency_matrix(out, sparse = F)
  return(out)
}, n = N, simplify = "array")

# rearrange
x_train <- grphs %>%
  array_reshape(c(train_size, N^2))


#......................
# train demon model
#......................
demon <- VariationalAutoEncoder(N^2, 64, 32)
optimizer <- optimizer_adam(learning_rate = 1e-3)
demon %>% compile(optimizer, loss = loss_mean_squared_error())
demon %>% fit(x_train, x_train, epochs = 2, batch_size = 2)


#......................
# predict from demon model
#......................
inputnets <- rbinom(n = 100, size = 1, prob = 0.05)
inputnets <- t(inputnets)
# new nets
new_nets <- demon$predict(inputnets)


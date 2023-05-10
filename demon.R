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



Latent_Loss <- function(codings_mean, codings_log_var){
  loss <- -0.5 * keras::k_sum(1 + codings_log_var - keras::k_exp(codings_log_var) - keras::k_square(codings_mean), axis=-1)
  return(loss)
}


#++++++++++++++++++++++++++++++++++++++++++
### VAE Demon Model Setup        ####
#++++++++++++++++++++++++++++++++++++++++++
#............................................................
# Project Specifics
#...........................................................
n <- 1e2 # population size, needs to be same from sim as nodes, etc
coding_dim <- 10

#............................................................
# Set up demon ENcoder
#...........................................................
encoder_inputs <- keras::layer_input(shape = c(n,n))
z <- keras::layer_flatten(object = encoder_inputs) %>%
  keras::layer_dense(., units = 150, activation = "selu") %>%
  keras::layer_dense(., units = 100, activation = "selu")
# mu - mean coding
codings_mean <- z %>%
  keras::layer_dense(coding_dim)
# lambda - log var coding
codings_log_var <- z %>%
  keras::layer_dense(coding_dim)
# generate new codings
codings <- CodingSampler()
codings <- codings(list(codings_mean, codings_log_var))

# make demon encoder now
demon_encoder <- keras::keras_model(inputs = encoder_inputs,
                                    outputs=list(codings_mean, codings_log_var, codings))
summary(demon_encoder)

#............................................................
# Set up demon DEcoder
#...........................................................
decoder_inputs <- keras::layer_input(shape = c(coding_dim))
# build back up
x <- decoder_inputs %>%
  keras::layer_dense(., units = 100, activation = "selu") %>%
  keras::layer_dense(., units = 150, activation = "selu") %>%
  keras::layer_dense(., units = n*n, activation="sigmoid")
# reshape
outputs <- x %>%
  keras::layer_reshape(., target_shape = c(n, n))
# make demon encoder now
demon_decoder <- keras::keras_model(inputs = decoder_inputs,
                                    outputs = outputs)
summary(demon_decoder)


#............................................................
# Set up demon combined EN-DEcoder
#...........................................................

codings <- demon_encoder(encoder_inputs)[[3]]
reconstructions <- demon_decoder(codings)
demon <- keras::keras_model(inputs = encoder_inputs,
                            outputs = list(codings_mean, codings_log_var, reconstructions))
summary(demon)



#++++++++++++++++++++++++++++++++++++++++++
### Training VAE Demon        ####
#++++++++++++++++++++++++++++++++++++++++++
#..............................
# generate training set
#...............................
train_size <- 5e3
rewireprob <- rnbinom(n = train_size, size = 5, mu = 25)/n
grphs <- lapply(rewireprob, function(x, n){
  out <- igraph::erdos.renyi.game(n = n,
                           p.or.m = x,
                           type = "gnp")
  out <- igraph::as_adjacency_matrix(out, sparse = F)
  return(out)
  }, n = n)
grphstf <- tensorflow::tf$convert_to_tensor(grphs)


demon %>% compile(
  optimizer = keras::optimizer_adam(learning_rate = 1e-3),  # Optimizer
  # Loss function to minimize
  loss = keras::loss_mean_squared_error()
)

demon %>%
  keras::fit(
    grphstf,
    grphstf,
    batch_size = 100,
    epochs = 50
  )

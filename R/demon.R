#' @title Coding Sampler for Demon VAE
#' @noMd
#' @noRd

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



#' @title Encoder for Demon VAE
#' @noMd
#' @noRd

Encoder(keras$layers$Layer) %py_class% {
  "Maps CNN to a triplet (z_mean, z_log_var, z)."

  initialize <- function(latent_dim, intermediate_dim,
                         name = "encoder", ...) {
    super$initialize(name = name, ...)
    self$codings_proj <- layer_dense(units = intermediate_dim,
                                     activation = "relu")
    self$additional_dense1 <- layer_dense(units = intermediate_dim,
                                         activation = "selu") #add layer 1
    self$dropout <- layer_dropout(rate = 0.5)  # Dropout layer
    self$additional_dense2 <-layer_dense(units = abs(intermediate_dim - 50),
                                         activation = "sigmoid") #add layer 2
    self$codings_mean <- layer_dense(units = latent_dim)
    self$codings_log_var <- layer_dense(units = latent_dim)
    self$sampling <- CodingSampler()
  }

  call <- function(inputs) {
    x <- self$codings_proj(inputs)
    x <- self$additional_dense1(x)
    x <-  self$dropout(x)
    x <- self$additional_dense2(x)
    codings_mean <- self$codings_mean(x)
    codings_log_var <- self$codings_log_var(x)
    codings <- self$sampling(c(codings_mean, codings_log_var))
    list(codings_mean, codings_log_var, codings)
  }
}

#' @title Decoder for Demon VAE
#' @noMd
#' @noRd

Decoder(keras$layers$Layer) %py_class% {
  "Converts z-codings, the encoded digit vector, back into a adjacency matrix."

  initialize <- function(original_dim,
                         intermediate_dim,
                         name = "decoder", ...) {
    super$initialize(name = name, ...)
    self$codings_proj <- keras::layer_dense(units = intermediate_dim,
                                            activation = "relu")
    self$additional_dense1 <- layer_dense(units = intermediate_dim,
                                          activation = "selu") #add layer 1
    self$dropout <- layer_dropout(rate = 0.5)  # Dropout layer
    self$additional_dense2 <-layer_dense(units = abs(intermediate_dim - 50),
                                         activation = "sigmoid") #add layer 2
    self$codings_output <- keras::layer_dense(units = original_dim,
                                              activation = "sigmoid")
  }

  call <- function(inputs) {
    x <- self$codings_proj(inputs)
    x <- self$additional_dense1(x)
    x <- self$dropout(x)
    x <- self$additional_dense2(x)
    self$codings_output(x)
  }
}

#' @title VAE: Combined Encoder-Decoder for full Demon
#' @param
#' @description
#' @details
#' @returns
#' @noMd
#' @noRd

DemonVariationalAutoEncoder(keras$Model) %py_class% {
  classname = "VAE: Combines the encoder and decoder into an end-to-end model"
  public = list(encoder = NULL,
                decoder = NULL,
                original_dim = NULL,
                intermediate_dim = NULL,
                latent_dim = NULL,

                initialize <- function(original_dim, intermediate_dim, latent_dim,
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
                  # fill in dims for public
                  self$original_dim <- original_dim
                  self$intermediate_dim <- intermediate_dim
                  self$latent_dim <- latent_dim

                }
  )

  call <- function(inputs) {
    c(codings_mean, codings_log_var, codings) %<-% self$encoder(inputs)
    reconstructed <- self$decoder(codings)
    # Add KL divergence regularization loss.
    kl_loss <- -0.5 * tf$reduce_mean(codings_log_var - tf$square(codings_mean) - tf$exp(codings_log_var) + 1)
    self$add_loss(kl_loss)
    reconstructed
  }
}


#' @title Decoder Gumbel Softmax for Demon VAE
#' @noMd
#' @noRd

DecoderGS(keras$layers$Layer) %py_class% {
  "Converts z-codings, the encoded digit vector, back into a adjacency matrix."

  initialize <- function(original_dim,
                         intermediate_dim,
                         softmax_temp,
                         hard,
                         name = "decoder", ...) {
    super$initialize(name = name, ...)
    self$codings_proj <- keras::layer_dense(units = intermediate_dim,
                                            activation = "relu")
    self$additional_dense1 <- layer_dense(units = intermediate_dim,
                                         activation = "selu") #add layer 1
    self$dropout <- layer_dropout(rate = 0.5)  # Dropout layer
    self$additional_dense2 <-layer_dense(units = abs(intermediate_dim - 50),
                                         activation = "sigmoid") #add layer 2
    self$codings_output <- keras::layer_dense(units = original_dim,
                                              activation = "linear") # need linear for logits --> Gumbell-Softmax; NB linear activation function is essentially a no-op, so output is just the weighted sum of the inputs plus the bias term aka "logits" as the raw outputs of a classification model.
    self$softmax_temp <- softmax_temp
    self$hard <- hard
  }

  call <- function(inputs) {
    x <- self$codings_proj(inputs)
    x <- self$additional_dense1(x)
    x <- self$dropout(x)
    x <- self$additional_dense2(x)
    logits <- self$codings_output(x) # These are now logits
    # Apply Gumbel-Softmax trick
    Uret <- tensorflow::tf$random$uniform(shape = tensorflow::tf$shape(logits), minval = 0, maxval = 1)
    gumbel_noise <- -tensorflow::tf$math$log(-tensorflow::tf$math$log(Uret + 1e-20) + 1e-20)
    gumbel_softmax_sample = tensorflow::tf$nn$softmax( (logits + gumbel_noise) /  self$softmax_temp )
    if (self$hard) {
      k = tf$shape(logits)[-1]
      gumbel_softmax_sample_hard <- tensorflow::tf$cast(tensorflow::tf$equal(gumbel_softmax_sample, tensorflow::tf$reduce_max(gumbel_softmax_sample, as.integer(1), keepdims = TRUE)), gumbel_softmax_sample$dtype)
      gumbel_softmax_sample <- tensorflow::tf$stop_gradient(gumbel_softmax_sample_hard - gumbel_softmax_sample) + gumbel_softmax_sample
    }
    gumbel_softmax_sample
  }
}


#' @title VAE Gumbel-Softmax: Combined Encoder-Decoder for full Demon
#' @param
#' @description
#' @details
#' @returns
#' @noMd
#' @noRd

DemonVariationalAutoEncoderGS(keras$Model) %py_class% {
  classname = "VAE Gumbel-Softmax: Combines the encoder and decoder into an end-to-end model"
  public = list(encoder = NULL,
                decoder = NULL,
                original_dim = NULL,
                intermediate_dim = NULL,
                latent_dim = NULL,
                softmax_temp = NULL,
                hard = NULL,

                initialize <- function(original_dim, intermediate_dim, latent_dim,
                                       softmax_temp, hard = TRUE,
                                       name = "autoencoder", ...) {
                  super$initialize(name = name, ...)
                  self$original_dim <- original_dim

                  self$encoder <- Encoder(
                    latent_dim = latent_dim,
                    intermediate_dim = intermediate_dim
                  )
                  self$decoder <- DecoderGS(
                    original_dim,
                    intermediate_dim = intermediate_dim,
                    softmax_temp = softmax_temp,
                    hard = hard
                  )
                  # fill in info for viewing
                  self$original_dim <- original_dim
                  self$intermediate_dim <- intermediate_dim
                  self$latent_dim <- latent_dim

                }
  )

  call <- function(inputs) {
    c(codings_mean, codings_log_var, codings) %<-% self$encoder(inputs)
    reconstructed <- self$decoder(codings)
    # Add KL divergence regularization loss.
    kl_loss <- -0.5 * tf$reduce_mean(codings_log_var - tf$square(codings_mean) - tf$exp(codings_log_var) + 1)
    self$add_loss(kl_loss)
    reconstructed
  }
}





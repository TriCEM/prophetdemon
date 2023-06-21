#' @title Prophet Forward CNN Model for Predicting Final Sizes
#' @param
#' @description
#' @details
#' @returns
#' @export

ProphetFCNN(keras$Model) %py_class% {
  initialize <- function(input_shape_dim, name = "ProphetModel") {
    super$initialize(name = name)
    self$model <- keras::keras_model_sequential() %>%
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
  }

  call <- function(inputs) {
    self$model(inputs)
  }
}

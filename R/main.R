# https://tensorflow.rstudio.com/guides/keras/making_new_layers_and_models_via_subclassing.html#putting-it-all-together-an-end-to-end-example
# https://tensorflow.rstudio.com/guides/keras/customizing_what_happens_in_fit.html#wrapping-up-an-end-to-end-gan-example
# https://tensorflow.rstudio.com/guides/keras/training_with_built_in_methods
# https://tensorflow.rstudio.com/guides/keras/making_new_layers_and_models_via_subclassing

#++++++++++++++++++++++++++++++++++++++++++
### Prophet        ####
#++++++++++++++++++++++++++++++++++++++++++
#' @title
#' @param
#' @description
#' @details
#' @returns
#' @export

prophet_model <- new_model_class(
  # name of model class
  classname = "prophet-sequentialNN",

    # Initialize Method for Prophet Sequential Class
  initialize = function(sequential_NNmodel) { # users will input layers manually
    # catch
    if (!any(class(sequential_NNmodel) %in% "keras.engine.sequential.Sequential") ) {
      stop("Model must be created with keras::keras_model_sequential")
    }
    super$initialize()
    self$sequential_NNmodel <- sequential_NNmodel
  },

  # Define compilation method
  compile = function(optimizer, loss_fn) {
    super$compile()
    self$optimizer <- optimizer
    self$loss_fn <- loss_fn
  },

  # define fit method
  fit = function(fit) {
    super$fit()
    self$fit <- fit
  }

  # define evaluation method

  # define predict method

)

#++++++++++++++++++++++++++++++++++++++++++
### Demon        ####
#++++++++++++++++++++++++++++++++++++++++++
#' @title
#' @param
#' @description
#' @details
#' @returns
#' @export

demon_model <- function() {
  #......................
  # checks
  #......................
  goodegg::

  #......................
  # setup (const, storage, etc)
  #......................


  #......................
  # core
  #......................


  #......................
  # out
  #......................
  return()
}


#++++++++++++++++++++++++++++++++++++++++++
### Duel        ####
#++++++++++++++++++++++++++++++++++++++++++
#' @title
#' @param
#' @description
#' @details
#' https://tensorflow.rstudio.com/guides/keras/customizing_what_happens_in_fit.html#wrapping-up-an-end-to-end-gan-example
#' https://github.com/ageron/handson-ml2
#' @returns
#' @export

faustduel <- ()

#' @title Q Function Approximator
#' @description
#' @returns FNN Keras Model
#' @export

create_Qfunx_approximator <- function() {
  model <- keras::keras_model_sequential() %>%
    keras::layer_dense(units = 24, activation = 'relu', input_shape = c(1)) %>%
    keras::layer_dense(units = 24, activation = 'relu') %>%
    keras::layer_dense(units = 2)  # two actions: train prophet or demon

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-3),
    loss = 'mse'
  )
  return(model)
}



#' @title
#' @param N integer; population size
#' @param fomes_reps integer; Number of fomes realizations to consider
#' @param fomes_beta numeric; Prob of infxn given contact
#' @param fomes_durI numeric; Duration of illness
#' @param fomes_rho numeric; Rewiring rate
#' @param prophet keras model; Prophet predictor
#' @param demon keras model; Demon VAE model
#' @param DQN keras model; Deep neural net use to estimate q-values
#' @param epsilon numeric: Probability of choosing a random action through epsilon greedy policy approach
#' @param gammafact Discount factor for future rewards via the credit assignment problem
#' @param batch_size: Batch size for when and size of memory replay that will be considered
#' @param epsilon numeric: Probability of choosing a random action through epsilon greedy policy approach
#' @param steps integer; Number of steps per epoch to be considered by the DQN
#' @param train_size integer; Number of training iterations to complete when fine-tuning the prophet or demon
#' @param trainwi numeric vector of length 2; Training weights for prophet and demon, respectively
#' @param demonprophet_epochs integer; Number of epochs to consider in prophet or demon retraining (i.e. fine-tuning)
#' @param demonprophet_btch_size integer; Batch size to consider in prophet or demon retraining (i.e fine-tuning)
#' @description Seesaw balancing: Prophet reward \deqn{P(x|data)} + Demon Reward \deqn{\frac{1}{var(F_s)}}
#' @details
#' @returns
#' @export

faustRL <- function(N, fomes_reps, fomes_beta, fomes_durI, fomes_rho,
                    prophet, demon, DQN,
                    epsilon = 0.1, gammafact = 0.95, batch_size = 64,
                    steps = 100,
                    trainwi = c(0.8,0.2), train_size = 100,
                    demonprophet_epochs = 10, demonprophet_btch_size = 32) {
  #............................................................
  # checks
  #............................................................
  goodegg::assert_single_int(N)
  goodegg::assert_single_int(fomes_reps)
  goodegg::assert_single_bounded(fomes_beta, left = 0, right = Inf)
  goodegg::assert_single_bounded(fomes_durI, left = 0, right = Inf)
  goodegg::assert_single_bounded(epsilon, left = 0, right = 1)
  goodegg::assert_single_bounded(gammafact, left = 0, right = 1)
  goodegg::assert_int(batch_size)
  goodegg::assert_numeric(trainwi)
  goodegg::assert_length(trainwi, 2)
  goodegg::assert_eq(sum(trainwi), 1, message = "Training weights must sum to 1")
  #goodegg::
  # need to check that prophet and demon produce correct structures
  # will have to class this ... and do class checks
  # check if DQN is appropriate too

  # Need to make sure this N and N from prophet and demon is the same
  #++++++++++++++++++++++++++++++++++++++++++
  ### Pieces for RL      ####
  #++++++++++++++++++++++++++++++++++++++++++
  #......................
  # empiric pdf
  # Uses kernel density estimation to estimate a log probability of the consistency of the fomes final sizes realizations and the prediction of final size from the prophet model
  #......................
  empiricPDF <- function(fs, fspred, N) {
    densfx <- stats::density(fs,
                             bw = 1, # individual counts
                             kernel = "epanechnikov",
                             from = 0, to = N)
    # Evaluate the approx PDF at a specific point
    out <- densfx$y[which.min(abs(densfx$x - fspred))] # use which.min to identify the index of the closest value, which is then taken from density y
    return(out/sum(densfx$y))
  }

  #......................
  # Prophet-Demon RL Balancing Equation
  #......................
  balance_prophet_demon <- function(fs, fspred, N, trainwi) {
    # checks (redundant since internal but for clarity)
    goodegg::assert_numeric(fs)
    goodegg::assert_numeric(fspred)
    goodegg::assert_single_int(N)
    goodegg::assert_vector(trainwi)
    goodegg::assert_length(trainwi, 2)
    goodegg::assert_eq(sum(trainwi), 1, message = "Training weights must sum to 1")
    # calculate fspred c/w fs fomes realizations
    fsaccupred <- empiricPDF(fs, fspred, N)
    # calculate variation of final sizes from fomes realizations
    # but will also need to normalize it so it is on same scale as our prob dist above
      fsvar <- var(fs)
    # maximum variance is:
    maxvar <- var( c(rep(1,floor(length(fs)/2)), rep(N, floor(length(fs)/2))) )
    normalized_fsvar = fsvar / maxvar # min variance is 0


    # balance eq
    return( fsaccupred*trainwi[1] + normalized_fsvar *trainwi[2] )
  }

  #...................
  # Setting up Memory for Replays
  #....................
  memory <- list()
  # Sample a batch of experiences from the memory
  sample_experiences <- function(batch_size) {
    indices <- sample(1:length(memory), size = batch_size)
    return(memory[indices])
  }

  #............................................................
  # Choosing Action - epsilon-greedy strategy
  #...........................................................
  choose_action <- function(model, state, epsilon) {
    # With probability epsilon choose a random action
    if (runif(1) < epsilon) {
      action <- sample(c("prophet", "demon"), 1)
    } else {
      # Otherwise choose the best action according to the current policy
      q_values <- predict(model, matrix(state, ncol = 1))
      action <- ifelse(which.max(q_values) == 1, "prophet", "demon")
    }
    # out
    return(action)
  }
  #............................................................
  # Execute Action and Get Reward
  #...........................................................
  execute_action_get_reward <- function(action, train_size,
                                        N, fomes_reps, fomes_beta, fomes_durI, fomes_rho) {
    # always need generate new networks
    newconmats <- tidyr::expand_grid(tr = 1:train_size,
                                     N = N) %>%
      dplyr::select(-c("tr")) %>%
      dplyr::mutate(init_contact_mat = purrr::map(N, generate_random_graph))
    #......................
    # spend time either FINE TUNING the prophet or the demon
    #......................
    if (action == "prophet") {
      # with our new networks, we now call fomes
      finetunetrainset <- tidyr::expand_grid(reps = 1:fomes_reps,
                                             beta = list(rep(fomes_beta, N)),
                                             dur_I = fomes_durI,
                                             rho = fomes_rho,
                                             newconmats,
                                             term_time = Inf) %>%
        dplyr::select(-c("reps")) %>%
        dplyr::mutate(fomesout = purrr::pmap(., fomes::sim_Gillespie_nSIR),
                      finalsize = purrr::map_dbl(fomesout, function(x){summary(x)$FinalEpidemicSize}),
                      finalsize = finalsize/N)
      # rearrange for tensorflow
      finetunetrainsetconmats <- tensorflow::as_tensor( lapply(finetunetrainset$init_contact_mat, as.matrix) )
      finetunetrainset_truefs <- tensorflow::as_tensor( finetunetrainset$finalsize  )
      finetunetrainset_truefs <- tensorflow::array_reshape(finetunetrainset_truefs, dim = c(nrow(finetunetrainset), 1))

      # update training to fine-tune prophet
      prophet %>%
        keras::fit( finetunetrainsetconmats, finetunetrainset_truefs,
                    epochs = demonprophet_epochs, batch_size = demonprophet_btch_size)

    } else if (action == "demon") {
      # train demon
      # rearrange for tensorflow
      finetunetrainsetconmats <- tensorflow::as_tensor( lapply(newconmats$init_contact_mat, as.matrix) )
      finetunetrainset <- finetunetrainsetconmats %>% tensorflow::array_reshape(c(train_size, N^2))
      # update training to fine-tune demon
      demon %>% keras::fit(finetunetrainset, finetunetrainset,
                           epochs = demonprophet_epochs, batch_size = demonprophet_btch_size)

    }
    #......................
    # Perform Adverserial
    # Calculate the reward: farther the way it is from zero, the more negative reward it feels
    #......................
    # pull latent dim from demon
    latent_dim <- demon$latent_dim
    # demon makes new network
    random_latent_vectors <- matrix(rnorm(1 * latent_dim), nrow = 1, ncol = latent_dim)
    inputnet <- sapply(as.numeric(demon$decoder(random_latent_vectors)), function(x){x > runif(1)})
    #TODO make this Gumbell Softmax
    inputnet <- matrix(inputnet, nrow = N, ncol = N)
    # TODO this is symmetry problem
    inputnet[lower.tri(inputnet)] <- t(inputnet)[lower.tri(inputnet)]
    # don't make demon care about diagonal
    diag(inputnet) <- 0

    # run fomes on it
    currfomes_fs <- tidyr::expand_grid(reps = 1:fomes_reps,
                                       N = N,
                                       beta = list(rep(fomes_beta, N)),
                                       dur_I = fomes_durI,
                                       init_contact_mat = list(inputnet),
                                       rho = fomes_rho,
                                       term_time = Inf) %>%
      dplyr::select(-c("reps")) %>%
      dplyr::mutate(fomesout = purrr::pmap(., fomes::sim_Gillespie_nSIR),
                    finalsize = purrr::map_dbl(fomesout, function(x){summary(x)$FinalEpidemicSize}),
                    finalsize = finalsize/N) %>%
      dplyr::pull(finalsize)

    # check our prediction
    inputnettens <- tensorflow::as_tensor(inputnet)
    inputnettens <- tensorflow::array_reshape(inputnet,
                                              c(1, N, N, 1))
    fspred <- as.numeric(prophet$predict(inputnettens))

    # initial state
    balance <- balance_prophet_demon(fs = currfomes_fs,
                                     fspred = fspred,
                                     N = N, trainwi = trainwi)

    # our max reward is 2
    reward <- -abs(balance - 2) #abs here in case kernel density misbehaves
    return(list(
      balance = balance,
      reward = reward
    ))
  }

  #++++++++++++++++++++++++++++++++++++++++++
  ###  Initialization of environment   ####
  #++++++++++++++++++++++++++++++++++++++++++
  # always start at mass action model, which should be base/simplest use case
  inputnet <- matrix(1, nrow = N, ncol = N)
  diag(inputnet) <- 0
  # run fomes
  currfomes_fs <- tidyr::expand_grid(reps = 1:fomes_reps,
                                     N = N,
                                     beta = list(rep(fomes_beta, N)),
                                     dur_I = fomes_durI,
                                     init_contact_mat = list(inputnet),
                                     rho = fomes_rho,
                                     term_time = Inf) %>%
    dplyr::select(-c("reps")) %>%
    dplyr::mutate(fomesout = purrr::pmap(., fomes::sim_Gillespie_nSIR),
                  finalsize = purrr::map_dbl(fomesout, function(x){summary(x)$FinalEpidemicSize}),
                  finalsize = finalsize/N) %>%
    dplyr::pull(finalsize)

  # initial prophet prediction
  inputnettens <- tensorflow::as_tensor(inputnet)
  inputnettens <- tensorflow::array_reshape(inputnet,
                                            c(1, N, N, 1))
  fspred <- as.numeric(prophet$predict(inputnettens))

  # initial state
  currstate <- balance_prophet_demon(fs = currfomes_fs,
                                     fspred = fspred,
                                     N = N, trainwi = trainwi)


  #++++++++++++++++++++++++++++++++++++++++++
  ###  Step Through Iterations       ####
  #++++++++++++++++++++++++++++++++++++++++++
  # set up new progress bar
  pb <- txtProgressBar(min = 1, max = steps, initial = 1, style = 3)
  # Start the episode
  for (step in 1:steps) {
    # call progress bar
    setTxtProgressBar(pb, step)
    # call out
    cat("Step: ", step)
    # Choose an action
    action <- choose_action(model = DQN, state = currstate, epsilon = epsilon)

    # Take the action and get the reward and new state
    ret <- execute_action_get_reward(action, train_size,
                                     N, fomes_reps, fomes_beta, fomes_durI, fomes_rho)
    next_state <- ret$balance
    reward <- ret$reward

    # Add the experience to the memory
    experience <- list(currstate = currstate, action = action, reward = reward, next_state = next_state)
    memory <- append(memory, list(experience))

    # Sample a batch of experiences from the memory and use it to update the model
    if (length(memory) >= batch_size) {
      experiences <- sample_experiences(batch_size)
      states <- sapply(experiences, function(x) x$state)
      actions <- sapply(experiences, function(x) x$action)
      rewards <- sapply(experiences, function(x) x$reward)
      next_states <- sapply(experiences, function(x) x$next_state)
      q_values_next <- DQN %>% predict(matrix(next_states, ncol = 1))
      targets <- rewards + gammafact * apply(q_values_next, 1, max)
      targets_full <- DQN %>% predict(matrix(states, ncol = 1))
      targets_full[cbind(1:length(actions), match(actions, c("prophet", "demon")))] <- targets
      DQN %>% fit(matrix(states, ncol = 1), targets_full, epochs = 1, verbose = 0)
    }

    # Update the state
    state <- next_state
  }
  # close progress bar
  close(pb)
  #............................................................
  # out
  # show how prophet and demon were trained
  #............................................................
  return(memory)
}


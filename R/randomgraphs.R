#' @title Generate Random Graph
#' @description From igraph functions, generate a random graph with no user
#' input
#' @returns contact adjacency matrix
#' @export

generate_random_graph <- function(N) {
  # pick random algorithm
  randlet <- sample(letters[1:6], 1)
  #......................
  # use switch to randomly pick an igraph random graph
  #......................
  switch(randlet,
         a = {
           # Erdős–Rényi Model
           rprob <- rexp(1, rate = 10)
           while(rprob > 1) {
             rprob <- rexp(1, rate = 10)
           }
           grph <- igraph::sample_gnp(n = N, p = rprob, directed = FALSE)
         },

         b = {
           # Barabási–Albert Model
           grph <- igraph::sample_pa(n = N, power = 1, m = 1, directed = FALSE)

         },
         c = {
           # Watts-Strogatz Model
           rnei <- sample(1:floor(sqrt(N)/2), 1)
           rprob <- rexp(1, rate = 10)
           while(rprob > 1) {
             rprob <- rexp(1, rate = 10)
           }
           grph <- igraph::sample_smallworld(dim = 1,
                                             size = N,
                                             nei = rnei, p = rprob)
         },

         d = {
           # Geometric Random Graph
           rradius <- runif(1)
           grph <- igraph::sample_grg( n = N, radius = rradius)
         },
         e = {
           # Forest Fire Model
           rprob <- rexp(1, rate = 10)
           while(rprob > 1) {
             rprob <- rexp(1, rate = 10)
           }
           rbw <- runif(1, 0, 0.25)
           grph <- igraph::sample_forestfire(n = N,
                                             fw.prob = rprob, bw.factor = rbw,
                                             directed = FALSE)
         },
         f = {
           #  Static Power Law Model
           rexpout <- runif(1, min = 2, max = 5)
           rexpin <- -runif(1, min = 2, max = 5)
           redges <- floor(N * runif(1, min = 2, max = 5))
           grph <- igraph::sample_fitness_pl(no.of.nodes = N,
                                     no.of.edges = redges,
                                     exponent.out = rexpout,
                                     exponent.in = rexpin,
                                     loops = FALSE, multiple = FALSE)
         }
  ) # end switch
  #......................
  # send out contact mat
  #......................
  conmat <- igraph::as_adjacency_matrix(grph, sparse = T)
  nrow(conmat)
  return(conmat)
}


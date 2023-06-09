"""
    Test ability to predict infection probs using discrete-time approximations
    Compare infection probs with stochastic-event driven sims under the Gillespie algorithm.
"""
import numpy as np
import json
import tensorflow as tf
import time
from tensorflow import keras

# terminate_condition = len(self.sample) < 50
global debug
debug = False

class SIRLayer(keras.layers.Layer):
    
    def __init__(self,nu,time_step,**kwargs):
        
        """
            Note: probably want to set trainable=False in kwargs
        """
        
        dtype = tf.float32 # data type (i.e float32 or float64)
        
        # Removal/recovery rate
        self.nu = nu
        
        # Time step used to numerically solve infection probs
        self.time_step = tf.constant(time_step,dtype=dtype)
        
        self.immunity = True
        
        super().__init__(**kwargs)
        
    def call(self, X):
        
        # Slice input tensor into epi variables
        beta = X[0:-2,:]
        prob_I = X[-2,:] 
        prob_R = X[-1,:] 
        
        # Compute individual-level probs of infection based on transmission rates in beta
        if self.immunity:
            pairs_SI = tf.tensordot(prob_I, 1.0 - prob_I - prob_R, axes=0) # np.outer(p,1-p-r)
        else:
            pairs_SI = tf.tensordot(prob_I, 1.0 - prob_I) # np.outer(p,1-p) -- outer product returns matrix with elements I_i * S_j
        betaSI = tf.multiply(beta,pairs_SI) # np.multiply(beta,pairs_SI) -- element-wise product returns matrix with elements B_i,j * S_j * I_i 
        trans_rates = tf.reduce_sum(betaSI,0) # np.sum(betaSI,0) -- sum over rows i.e. potential infectors
        trans_probs = 1 - tf.exp(-trans_rates * self.time_step) # convert to infection prob
        
        # Compute individual-level recovery probs
        recov_rates = self.nu * prob_I
        recov_probs = 1 - tf.exp(-recov_rates * self.time_step)
        
        # Update variables
        prob_I = prob_I + trans_probs - recov_probs
        if self.immunity:
           prob_R = prob_R + recov_probs 

        prob_I = tf.reshape(prob_I, [1,n])
        prob_R = tf.reshape(prob_R, [1,n])
        return tf.concat([beta, prob_I, prob_R], 0)
    
class SIRModel:
    
    """
        Should this extend keras.Model?
        Could also write a call function that takes beta/contact matrix as
        input so we can just call the model to run the epi model
    """
    
    def __init__(self,**kwargs):
        
        dtype = tf.float32 # data type (i.e float32 or float64)
        
        init_I = kwargs.get('init_I', [1,0,0,0])
        self.prob_I = tf.Variable([init_I],dtype=dtype,trainable=False)
        
        init_R = kwargs.get('init_R', np.zeros(len(init_I)))
        self.prob_R = tf.Variable([init_R],dtype=dtype,trainable=False)
        
        # Time at which to end sim -- should be consistent terminate_condition
        self.final_time = kwargs.get('final_time', 5.0)
            
        # Transmission rate(s) between pops
        beta = kwargs.get('beta', None)
        self.beta = tf.constant(beta,dtype=dtype)
        
        # Removal/recovery rate
        nu = kwargs.get('nu', 1.)
        self.nu = tf.constant(nu,dtype=dtype)
        
        # Immune waining rate -- set to np.inf for SIS model OR zero for SIR model
        self.immunity = kwargs.get('immunity', True)
        
        # Time step used to numerically solve infection probs
        time_step = kwargs.get('time_step', 0.1)
        self.time_step = time_step
        
        # Compose model from SIR Layers
        n_steps = int(self.final_time / self.time_step)
        
        # Input tensor
        #X = tf.concat([self.beta, self.prob_I, self.prob_R], 0)
        
        self.model = keras.Sequential()
        #epi_model.add(keras.Input(shape=X.shape))
        #epi_model.summary()
        
        # Test single layer op - this way does not seem to work
        #input_layer = SIRLayer(nu,time_step,input_shape=X.shape)
        #slay1 = SIRLayer(nu,time_step,input_shape=X.shape)
        #slay2 = SIRLayer(nu,time_step,input_shape=X.shape)
        
        #But not specifying input shape does?
        #slay1 = SIRLayer(nu,time_step)
        #slay2 = SIRLayer(nu,time_step)
        
        for _ in range(n_steps):
            self.model.add(SIRLayer(nu,time_step,trainable=False))
        
        final_layer = keras.layers.Lambda(lambda X: tf.reduce_sum(X[-2,:] + X[-1,:])) # prob_I + prob_R
        self.model.add(final_layer)

        # To use
        #epi_model.summary()
        #epi_model.layers()
        #epi_model.inputs()
        #epi_model.outputs()
        
        #final_size = epi_model(X)
        #print(final_size.numpy())
        
        #return epi_model
    
    def call(self, input_beta):
        
        X = tf.concat([input_beta, self.prob_I, self.prob_R], 0)
        final_size = self.model(X)
        return final_size

class SIREmulator:
    
    """
        Original TF implementation of SIR CTMC model
        No longer needed so can eventually delete
    """
    
    def __init__(self,**kwargs):
        
        dtype = tf.float32 # data type (i.e float32 or float64)
        
        init_I = kwargs.get('init_I', [1,0,0,0])
        self.prob_I = tf.Variable(init_I,dtype=dtype)
        
        init_R = kwargs.get('init_R', np.zeros(len(init_I)))
        self.prob_R = tf.Variable(init_R,dtype=dtype)
        
        # Time at which to end sim -- should be consistent terminate_condition
        self.final_time = kwargs.get('final_time', 5.0)
            
        # Transmission rate(s) between pops
        beta = kwargs.get('beta', None)
        self.beta = tf.constant(beta,dtype=dtype)
        
        # Removal/recovery rate
        nu = kwargs.get('nu', 1.)
        self.nu = tf.constant(nu,dtype=dtype)
        
        # Immune waining rate -- set to np.inf for SIS model OR zero for SIR model
        self.immunity = kwargs.get('immunity', True)
        
        # Time step used to numerically solve infection probs
        self.time_step = kwargs.get('time_step', 0.1)
    
    def run(self):
        
        """
            Approximate final-size distribution of epidemic by
            tracking individual probabilities that each node has been infected
            Note: Assumes SIS model unless immunity=True for a SIR model
        """

        beta = self.beta
        nu = self.nu
        prob_I = self.prob_I # individual probabilities of being infected 
        prob_R = self.prob_R # individual probabilities of being recovered
        time = 0
        final_time = self.final_time
        time_step = self.time_step
        while time < final_time:
            
            # Compute individual-level probs of infection based on transmission rates in beta
            if self.immunity:
                pairs_SI = tf.tensordot(prob_I, 1.0 - prob_I - prob_R, axes=0) # np.outer(p,1-p-r)
            else:
                pairs_SI = tf.tensordot(prob_I, 1.0 - prob_I) # np.outer(p,1-p) -- outer product returns matrix with elements I_i * S_j
            betaSI = tf.multiply(beta,pairs_SI) # np.multiply(beta,pairs_SI) -- element-wise product returns matrix with elements B_i,j * S_j * I_i 
            trans_rates = tf.reduce_sum(betaSI,0) # np.sum(betaSI,0) -- sum over rows i.e. potential infectors
            trans_probs = 1 - tf.exp(-trans_rates * time_step) # convert to infection prob
            
            # Compute individual-level recovery probs
            recov_rates = nu * prob_I
            recov_probs = 1 - tf.exp(-recov_rates * time_step)
            
            # Update variables
            prob_I = prob_I + trans_probs - recov_probs
            if self.immunity:
               prob_R = prob_R + recov_probs 
            time += time_step
        
        p_infected = prob_I + prob_R # prob of ever being infected (only works for SIR model)
        #mean_size = np.sum(p_infected)
        
        return p_infected
 
class GillespieSim():
    
    """
        Simulate stochastic event-driven epi dynamics
        Note: assumes SIS dynamcics for now
    """
    
    
    def __init__(self,**kwargs):
        
        self.init_I = kwargs.get('init_I', [1,0,0,0])
        
        self.pops = kwargs.get('pops', len(self.init_I)) # pops = individuals for network model
        
        self.init_S = kwargs.get('init_S', 1-self.init_I)
        
        self.init_R = kwargs.get('init_R', np.zeros(self.pops))
        
        # Condition to terminate sim
        self.terminate_condition = kwargs.get('terminate_condition',"t[i] > 5")
        
        # Time at which to end sim -- should be consistent terminate_condition
        self.final_time = kwargs.get('final_time', 5.0)
            
        # Transmission rate(s) between pops
        random_beta = np.random.uniform(low=0.0, high=1.0, size=[self.pops,self.pops])
        np.fill_diagonal(random_beta, 0.)
        self.beta = kwargs.get('beta', random_beta)
            
        # Removal/recovery rate
        self.d = kwargs.get('d', 1.)
        
        # Immune waining rate -- set to np.inf for SIS model OR zero for SIR model
        self.omega = kwargs.get('omega', 0.0)
        
        # Json traj file
        self.traj_file = kwargs.get('traj_file',"test.json")
        
        self.time_bins = kwargs.get('time_bins',1)
        
        self.time_intervals = []
        
        self.write_trajectories = kwargs.get('write_trajectories',False)

    def run(self):
        
        """
            Run epi simulation using Gillespie algorithm
        """
        
        # Unpack params from self.params dict and convert to numpy arrays
        pops = self.pops
        terminate_condition = self.terminate_condition
        final_time = self.final_time
        beta = np.array(self.beta)
        d = np.array(self.d)
        omega = np.array(self.omega)
    
        # Keep numpy arrays for counts
        S_now = np.array(self.init_S)
        I_now = np.array(self.init_I)
        R_now = np.array(self.init_R)

        N_now = S_now + I_now + R_now
    
        # initialize time tracker and index counter
        time = 0.0
        t = [0]
        i = 0
    
        # Event trackers
        S_traj = []
        I_traj = []
        R_traj = []
        for p in range(pops):
            S_traj.append([])
            I_traj.append([])
            R_traj.append([])
        
        total_infections = np.sum(I_now)
        complete = False
    
        # if debug: print(f"will terminate when {terminate_condition}") # Terminate Condition
        while not eval(terminate_condition):
    
            # If no one is infected, exit while loop
            if np.sum(I_now) == 0: # or np.sum(S_now) == 0:
                break
    
            # Transmission rates
            #betaSI = np.multiply(beta,np.outer(I_now,np.ones(pops)))
            betaSI = np.multiply(beta,np.outer(I_now,S_now)) # so betaSI has elements beta_i,j * S_j * I_i
            R_t = np.sum(betaSI) #beta * self.I_pop.size
            
            # Recovery rates
            dI = d * I_now
            R_r = np.sum(dI)
            
            # Immune waining
            if np.any(np.isfinite(omega)):
                R_w = np.sum(omega * R_now)
            else:
                R_w = 0
    
            # Calculate time until each of the events occurs
            events = {}
            
            if R_t > 0:
                events['transmission'] = np.random.exponential((1 / R_t))
            else:
                events['transmission'] = np.inf
                 
            if R_r > 0:    
                events['recovery'] = np.random.exponential((1 / R_r))
            else:
                events['recovery'] = np.inf
            if R_w > 0:
                events['waining'] = np.random.exponential((1 / R_w))
            else:
                events['waining'] = np.inf
    
            # Get the event that will occur first, and the time that will take
            next_event = min(events, key=lambda key: events[key])
    
            # Jump to that time
            time = t[i] + events[next_event]
            
            if time > final_time:
                complete = True
                break
    
            if debug:
                print(f"# susceptible: {S_now}  # infected: {I_now})  # recovered: {R_now}")
                #print(f"Population Beta: {round(beta,3)}  Time to transmission (R_t): {round(events['transmission'],3)}  Time to recovery (R_r): {round(events['recovery'],3)}, Time to mutation: {round(events['mutation'], 3)}")
            
            if 'transmission' == next_event:
                if debug: print("transmission")
                
                #Choose infector pop based on transmission rates
                pop_probs = np.sum(betaSI,1) / R_t # sum is over the columns (i.e. the S_pops)
                parent_pop = np.random.choice(pops, 1, p=pop_probs)[0]
                
                pop_probs = betaSI[parent_pop] / np.sum(betaSI[parent_pop])
                child_pop = np.random.choice(pops, 1, p=pop_probs)[0]
                
                # population level stuff
                S_now[child_pop] -= 1
                I_now[child_pop] += 1 # had parent_pop (incorrectly) here for some reason?
                
                total_infections += 1
                
            # If recovery occurs first
            elif 'recovery' == next_event:
                if debug: print("recovery")
                
                # Choose recovery pop based on recovery rates
                pop_probs = dI / R_r
                recoveree_pop = np.random.choice(pops, 1, p=pop_probs)[0]
                    
                # population level stuff
                I_now[recoveree_pop] -= 1
                if np.isinf(omega):
                    S_now[recoveree_pop] += 1
                else:
                    R_now[recoveree_pop] += 1
                    
                    
            elif 'waining' == next_event:
                if debug: print("waining")
                
                #pop_probs = omega * R_now / np.sum(omega * R_now)
                
                """
                    Never actually implemented immune waining
                    As of now we assume omega = 0 or omega = Inf
                """
    
            # Update lists and counters
            t.append(time)
            i += 1
            for p in range(pops):
                S_traj[p].append(int(S_now[p]))
                I_traj[p].append(int(I_now[p]))
                R_traj[p].append(int(R_now[p]))
    
            if debug:
                print("\n")
            #else:
                # TODO: Print sample population size counter also
                if i%200==0: print("t:", time)        
            
        """
            Dump pop trajectories into JSON file
        """
        if self.write_trajectories:
            traj_dict = {
                "S" : S_traj,
                "I" : I_traj,
                "R" : R_traj,
                "t" : t
            }
            with open(self.traj_file, "w") as outfile:
                json.dump(traj_dict, outfile)
           
        return complete, total_infections

if __name__ == '__main__':

    # Init configuration of infected individuals on network
    n = 100 # pop size
    final_time = 5.0 # time simulations should end
    init_I = np.zeros(n)
    init_I[0] = 1 # seed first infection
    
    # Populate a random connectivity/transmission rate matrix representing the contact network
    random_beta = np.random.uniform(low=0.0, high=0.02, size=[n,n])
    np.fill_diagonal(random_beta, 0.)
    
    nu = 0.5 # recovery rate
    
    """
        Try composing model with untrainable epi layers
    """
    epi_model = SIRModel(final_time=final_time,
                        init_I=init_I,
                        beta=random_beta,
                        immunity=True, 
                        nu=nu,
                        time_step=0.1)
    
    
    # Approximate infections probs
    # emulator = SIREmulator(final_time=final_time,
    #                     init_I=init_I,
    #                     beta=random_beta,
    #                     immunity=True, 
    #                     nu=nu,
    #                     time_step=0.05)
    
    tic = time.perf_counter()
    final_size = epi_model.call(random_beta)
    toc = time.perf_counter()
    elapsed = toc - tic

    print(f"Elapsed time: {elapsed:0.4f} seconds")
    print("Average final size:" + str(final_size.numpy()))

    # #Init stochastic simulation -- variables not supplied as keyword args will default to their defined values in __init__
    # sim = GillespieSim(pops=n,
    #                     terminate_condition="t[i] > 5",
    #                     final_time=final_time,
    #                     init_I=init_I,
    #                     beta=random_beta,
    #                     d=[nu],
    #                     omega=0.)
    
    # # Run stochastic sims to compare exact probs to approx probs
    # n_sims = 500
    # final_infections = []
    # for s in range(n_sims):
    #     print("Sim # :" + str(s))
    #     complete = False
    #     while not complete: # make sure sims completes to final time
    #         complete, final_size = sim.run()
    #     final_infections.append(final_size)
    

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # png_file = 'final_size_dist_gillespie.png' # + f'{param:.2f}' +'.png'
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # sns.histplot(final_infections,discrete=True,kde=False,stat='density')
    # ax.set_xlabel('Final size')
    # ax.set_ylabel('Frequency')
    # fig.tight_layout()
    # fig.savefig(png_file, dpi=200)
    

    
    
           
    
    
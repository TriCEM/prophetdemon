"""
    Test ability to predict infection probs using discrete-time approximations
    Compare infection probs with stochastic-event driven sims under the Gillespie algorithm.
"""
import numpy as np
import random
import json
import time
from poibin import PoiBin

# terminate_condition = len(self.sample) < 50
global debug
debug = False
 
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

def approx_infection_probs(init_I,beta,nu,final_time=5.0,time_step=0.1):
    
    """
        Predict individual infection probs using a discrete-time approximation
        Note: this assumes SIS dynamics for now
    """
    
    p_never_traj = []
    
    p = init_I # individual probabilities of being infected 
    time = 0
    final_time = 5.
    while time < final_time:
        
        # Compute individual-level probs of infection based on transmission rates in beta
        pairs_SI = np.outer(p,1-p) # outer product returns matrix with elements I_i * S_j
        betaSI = np.multiply(beta,pairs_SI) # element-wise product returns matrix with elements B_i,j * S_j * I_i 
        trans_rates = np.sum(betaSI,0) # sum over rows i.e. potential infectors
        trans_probs = 1 - np.exp(-trans_rates * time_step) # convert to infection prob
        
        # Compute individual-level recovery probs
        recov_rates = nu * p
        recov_probs = 1 - np.exp(-recov_rates * time_step)
        
        # Update variables
        p = p + trans_probs - recov_probs
        time += time_step
    
    return p

def approx_final_size_dist(init_I,beta,nu,final_time=5.0,time_step=0.1,immunity=True):
    
    """
        Approximate final-size distribution of epidemic by
        tracking individual probabilities that each node has never been infected
        Then treat probs of never being infected as independent Bernoulli trials
        and compute final size distribution using DFT approximation to the Poisson-Binomial distribution 
        Note: Assumes SIS model unless immunity=True for a SIR model
    """
    
    #p_never_traj = []
    
    p = init_I # individual probabilities of being infected 
    p_never = 1 - p # prob of never having been infected
    r = np.zeros(n) # individual probabilities of being recovered
    time = 0
    final_time = 5.
    while time < final_time:
        
        # Compute individual-level probs of infection based on transmission rates in beta
        if immunity:
            pairs_SI = np.outer(p,1-p-r)
        else:
            pairs_SI = np.outer(p,1-p) # outer product returns matrix with elements I_i * S_j
        betaSI = np.multiply(beta,pairs_SI) # element-wise product returns matrix with elements B_i,j * S_j * I_i 
        trans_rates = np.sum(betaSI,0) # sum over rows i.e. potential infectors
        trans_probs = 1 - np.exp(-trans_rates * time_step) # convert to infection prob
        
        # Update cumulative prob of never having been infected
        #p_never = p_never * (1 - trans_probs)
        #p_never_traj.append(p_never)
        
        # Compute individual-level recovery probs
        recov_rates = nu * p
        recov_probs = 1 - np.exp(-recov_rates * time_step)
        
        # Update variables
        p = p + trans_probs - recov_probs
        if immunity:
           r = r + recov_probs 
        time += time_step
    
    #pb = PoiBin(1-p_never)
    p_infected = p + r # prob of ever being infected (only works for SIR model)
    pb = PoiBin(p_infected)
    x = np.arange(0,len(p_infected)+1)
    fsd = pb.pmf(x) # final size dist
    
    return fsd

if __name__ == '__main__':

    # Init configuration of infected individuals on network
    n = 100 # pop size
    final_time = 5.0 # time simulations should end
    init_I = np.zeros(n)
    init_I[0] = 1 # seed first infection
    nu = 0.5 # recovery rate
    time_step = 0.01
    
    # Populate a random connectivity/transmission rate matrix representing the contact network
    random_beta = np.random.uniform(low=0.0, high=0.02, size=[n,n])
    np.fill_diagonal(random_beta, 0.)
    
    # Approximate infections probs
    #tic = time.perf_counter()
    #approx_inf_probs = approx_infection_probs(init_I, random_beta, nu, final_time=final_time, time_step=time_step)
    #toc = time.perf_counter()
    #elapsed = toc - tic
    #print(f"Elapsed time: {elapsed:0.4f} seconds")
    
    # Approximate infections probs
    tic = time.perf_counter()
    fsd = approx_final_size_dist(init_I, random_beta, nu, final_time=final_time, time_step=time_step, immunity=True)
    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"Elapsed time: {elapsed:0.4f} seconds")
    print('Final size distribution: ' + str(fsd))
    
    
    # Init stochastic simulation -- variables not supplied as keyword args will default to their defined values in __init__
    sim = GillespieSim(pops=n,
                        terminate_condition="t[i] > 5",
                        final_time=final_time,
                        init_I=init_I,
                        beta=random_beta,
                        d=[nu],
                        omega=0.)
    
    # Run stochastic sims to compare exact probs to approx probs
    n_sims = 1000
    final_infections = []
    for s in range(n_sims):
        print("Sim # :" + str(s))
        complete = False
        while not complete: # make sure sims completes to final time
            complete, final_size = sim.run()
        final_infections.append(final_size)
    

    import matplotlib.pyplot as plt
    import seaborn as sns
    png_file = 'final_size_dist_vs_poibin-approx.png' # + f'{param:.2f}' +'.png'
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    #sns.distplot(final_infections,hist = True, norm_hist = True, bins = 11)
    sns.histplot(final_infections,discrete=True,kde=False,stat='density')
    sns.lineplot(x=np.arange(0,n+1), y=fsd, color='orange')
    ax.set_xlabel('Final size')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig(png_file, dpi=200)
    
    
           
    
    
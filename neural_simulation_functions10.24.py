#!/usr/bin/env python
# coding: utf-8

# In[1]:


import brian2 as b2 # used for neural simulation
from brian2 import *

def independent_poisson_processes(num_neurons, time, num_samples, rate = None):
    """
    Description:
    -----------
    Generates independent Poisson process spike trains for multiple neurons.
    
    Parameters:
    -----------
    num_neurons : Number of neurons to simulate
    rate : Firing rate of neurons in Hz
    time : Duration of simulation in milliseconds
    num_samples : Number of samples to generate for each neuron
    
    Returns:
    -------
    spike_train : returns the spike event and timing for each neuron
    
    Note:
    -----
    - Uses two for loops so I can iterate each neuron and their samples.
    - I want to implement read() also but just putting read() is causing an error
    - Probably because I need it to read to something instead of just read()
    """
    spike_trains = []
    P = PoissonGroup(num_neurons, rate * b2.Hz)
    spike_monitor = SpikeMonitor(P)
    defaultclock.dt = 0.1 * ms
    
    store()

    for sample in range(num_samples):
        restore()
        run(time * ms)
            
        spike_trains_dict = spike_monitor.spike_trains() # this is the guy that makes ms and seconds in the output
        spike_trains.append(spike_trains_dict)
    
    return spike_trains


# In[2]:


def multivariate_constant_poisson(num_neurons, time, num_samples, rates = None):
    """
    Description:
    -----------
    Generates Poisson process spike trains with different constant rates per neuron.
    
    Parameters:
    -----------
    num_neurons : Number of neurons to simulate
    rates_array : Array of firing rates for each neuron in Hz (length = num_neurons)
    time : Duration of simulation in milliseconds
    num_samples : Number of samples to generate for each neuron
    
    Returns:
    -------
    spike_train : returns the spike event and timing for each neuron
    
    Note:
    -----
    - This tests heteroscedastic behavior (different variances per neuron)
    """
    if len(rates) != num_neurons:
        raise ValueError(f"Length of rates ({len(rates)}) must equal num_neurons ({num_neurons})")
    
    spike_trains = []
  # P = PoissonGroup(num_neurons, rate * b2.Hz) for singular
    P = PoissonGroup(num_neurons, rates * b2.Hz)
    spike_monitor = SpikeMonitor(P)
    defaultclock.dt = 0.1 * ms
    
    store()

    for sample in range(num_samples):
        restore()
        run(time * ms)
            
        spike_trains_dict = spike_monitor.spike_trains()
        spike_trains.append(spike_trains_dict)
    
    return spike_trains


# In[3]:


def generate_perturbed_rates(num_neurons, base_rate, perturbation_magnitude, relative=True):
    """
    Generate perturbed rates for Poisson processes.
    
    Parameters:
    -----------
    num_neurons : int
        Number of neurons
    base_rate : float or array
        Base firing rate(s) in Hz. If float, same rate for all neurons.
        If array, different base rate per neuron.
    perturbation_magnitude : float
        If absolute=True: Standard deviation of perturbation as fraction of base_rate
        If absolute=False: Absolute standard deviation of perturbation in Hz
    absolute : bool, default=True
        If True, perturbation_magnitude is absolute to base_rate (fraction)
        If False, perturbation_magnitude is relative value in Hz
    
    Returns:
    --------
    perturbed_rates : numpy.ndarray
        Contains perturbed rates for each neuron
    """
    
    # Handle single rate vs per-neuron rates
    if np.isscalar(base_rate):
        base_rates = np.full(num_neurons, base_rate) # np.full will make an array full of base_rate the length of num neurons
    else:
        base_rates = np.array(base_rate)
        if len(base_rates) != num_neurons:
            raise ValueError(f"Length of base_rate ({len(base_rates)}) must equal num_neurons ({num_neurons})")
    
    # Generate perturbations
    perturbed_rates = np.zeros((num_neurons, num_neurons))
    
    for neuron in range(num_neurons):
        if relative:
            # Relative mode: perturbation magnitude with rate
            std_dev = perturbation_magnitude * base_rates
        else:
            # Absolute mode: perturbation magnitude without rate
            std_dev = perturbation_magnitude
        
        # Generate perturbations with mean=0 and calculated std_dev
        perturbations = np.random.normal(0, std_dev, num_neurons)
        perturbed_rates = np.maximum(base_rates + perturbations, 0.1 * base_rates)
    
    return perturbed_rates


# In[4]:


def count_spikes(binary_data):
    """
    Counts cumulative spikes over time for multiple neurons across multiple samples.
    
    Parameters:
    -----------
    binary_data : List of lists containing numpy arrays of binary spike data
                 where 1 indicates a spike and 0 indicates no spike
    
    Returns:
    --------
    counts : List of lists containing numpy arrays of cumulative spike counts
    """
    counts = []
    
    # For each sample
    for sample in binary_data:
        sample_counts = []
        
        # For each neuron in the sample
        for neuron_data in sample:
            # Use cumsum to count spikes cumulatively
            spike_count = np.cumsum(neuron_data)
            sample_counts.append(spike_count)
            
        counts.append(sample_counts)
    
    return counts


# In[5]:


def calculate_theoretical_mean(rate, time):
    n = rate.shape[0]
    theoretical_mean = np.zeros((n, time))
    for i in range(n):
        #theoretical_mean[i]= (rate[i] * range(time)) 
        theoretical_mean[i] = (rate[i] / time) * np.arange(time)
    return theoretical_mean
    # rate is a vector, same size as the number of neurons. rate = 
    # needs to not just be for two neurons, make it for however neurons there are
    # rate_i times time, where i is the neuron 
    # rate should be an array


# In[6]:


def calculate_theoretical_std_dev(theoretical_means):
    return np.sqrt(theoretical_means)


# In[7]:


def empirical_means(count):
    total = 0

    for sample in count:
        total += sample
        print(sample)

    # np.mean(total) test this with the list of samples
    mean = total / num_samples
    return mean


# In[8]:


def std_dev_empirical_mean(empirical_mean, num_samples, count):
    # Initialize the sum of squared differences
    sum_squared_diff = 0

    # Calculate the sum of squared differences
    for sample in count:
        sum_squared_diff += (sample - empirical_mean) ** 2

    # Calculate the variance
    variance = sum_squared_diff / (num_samples - 1)

    # Calculate the standard deviation
    std_dev = np.sqrt(variance)

    return std_dev


# In[9]:


def variance_of_residuals(observed, expected):
    residuals = observed - expected
    return np.var(residuals, ddof=1)  # ddof=1 for sample variance


# In[10]:


def plot_count_neuron1_vs_time(count, num_samples, title = 'title'):
    plt.figure(figsize=(10,6))
    for i in range(num_samples):
        plt.plot(count[i][0], label=f'Sample {i}')
    #mean_neuron1 = np.mean([counting_process_nd[i][0] for i in range(num_samples)], axis=0)
    #plt.plot(mean_neuron1, label='Mean', color='black', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Count of Neuron 1')
    plt.title(title)
    plt.ylim(0, None)  # Set y-axis lower limit to 0
    plt.show()


# In[11]:


def plot_count_neuron1_vs_neuron2_vs_time(count, num_samples, neuron1_idx=0, neuron2_idx=1, title='title'):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_samples):
        ax.plot(range(len(count[i][neuron1_idx])), count[i][neuron1_idx], count[i][neuron2_idx], )
    #mean_neuron1 = np.mean([counting_process_nd[i][0] for i in range(num_samples)], axis=0)
    #mean_neuron2 = np.mean([counting_process_nd[i][1] for i in range(num_samples)], axis=0)
    #ax.plot(mean_neuron1, mean_neuron2, range(len(mean_neuron1)), color='black', linewidth=2)
    ax.set_xlabel('Time', fontsize = 14)
    ax.set_ylabel(f'Count of Neuron {neuron1_idx}', fontsize=14)
    ax.set_zlabel(f'Count of Neuron {neuron2_idx}', rotation=90, fontsize=14)
    ax.set_title(title, fontsize = 16)
    ax.set_xlim(0, max([len(count[i][neuron1_idx]) for i in range(num_samples)]))
    ax.set_ylim(0, max([max(count[i][neuron1_idx]) for i in range(num_samples)]))
    ax.set_zlim(0, max([max(count[i][neuron2_idx]) for i in range(num_samples)]))
    plt.show()    


# In[12]:


def plot_centered_staircase_3d(count, theoretical_means, theoretical_std_dev, num_samples, neuron1_idx=0, neuron2_idx=1, title = 'title'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Iterate through the number of samples
    for i in range(num_samples):
        # Ensure that counting_process_nd_result[i] is structured correctly
        neuron1_data = count[i][neuron1_idx] - theoretical_means[neuron1_idx]
        neuron2_data = count[i][neuron2_idx] - theoretical_means[neuron2_idx]
        
        # Plot the data, by subtracting by the theoretical_means, i should be centering the data but it is not doing that ....
        ax.plot(range(len(neuron1_data)), neuron2_data, neuron1_data)

    # Calculate mean centered data for both neurons 
#    mean_centered_neuron1 = np.mean([(counting_process_nd_result[i][0] - theoretical_means[0]) / theoretical_std_dev[0] for i in range(num_samples)], axis=0)
#    mean_centered_neuron2 = np.mean([(counting_process_nd_result[i][1] - theoretical_means[1]) / theoretical_std_dev[1] for i in range(num_samples)], axis=0)

    # Plot mean centered data
#    ax.plot(range(len(mean_centered_neuron1)), mean_centered_neuron2, mean_centered_neuron1, color='black', linewidth=2)

    ax.set_xlabel('Time', fontsize = 14)
    ax.set_ylabel(f'Count of Neuron {neuron1_idx}', fontsize=14)
    ax.set_zlabel(f'Count of Neuron {neuron2_idx}', rotation=90, fontsize=14)
    ax.set_title(title, fontsize = 16)

    ax.set_xlim(0, max(len(count[i][neuron1_idx]) for i in range(num_samples)))
#    max_val = max(max((counting_process_nd_result[i][1] - theoretical_means[1]) / theoretical_std_dev[1]) for i in range(num_samples))
#    ax.set_ylim(-max_val, max_val)
#    max_val = max(max((counting_process_nd_result[i][0] - theoretical_means[0]) / theoretical_std_dev[0]) for i in range(num_samples))
#    ax.set_zlim(-max_val, max_val)

    # Calculate the range of all data to make axes equal
    all_neuron1_data = [count[i][neuron1_idx] - theoretical_means[neuron1_idx] for i in range(num_samples)]
    all_neuron2_data = [count[i][neuron2_idx] - theoretical_means[neuron2_idx] for i in range(num_samples)]
    
    # Find the maximum absolute value across both neurons for equal scaling
    max_neuron1 = max(max(np.abs(data)) for data in all_neuron1_data)
    max_neuron2 = max(max(np.abs(data)) for data in all_neuron2_data)
    max_val = max(max_neuron1, max_neuron2)
    
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    
    ax.set_box_aspect([1,1,1])

    plt.show()


# In[13]:


def plot_standardized_staircase_3d(count, theoretical_means, theoretical_std_dev, t0, num_samples, neuron1_idx=0, neuron2_idx=1, title = 'title'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through the number of samples
    for i in range(num_samples):
        # Ensure that counting_process_nd_result[i] is structured correctly
        neuron1_data = (count[i][neuron1_idx] - theoretical_means[neuron1_idx]) / (theoretical_std_dev[neuron1_idx])
        neuron2_data = (count[i][neuron2_idx] - theoretical_means[neuron2_idx]) / (theoretical_std_dev[neuron2_idx])
        
        # Plot the data, len(neuron1_data) should be = to len(neuron2_data) so either can be used
        ax.plot(range(t0,len(neuron1_data)), neuron2_data[t0:], neuron1_data[t0:])

    # Calculate mean standardized data for both neurons
#    mean_standardized_neuron1 = np.mean([counting_process_nd_result[i][0] / theoretical_std_dev[0] for i in range(num_samples)], axis=0)
#    mean_standardized_neuron2 = np.mean([counting_process_nd_result[i][1] / theoretical_std_dev[1] for i in range(num_samples)], axis=0)

    # Plot mean standardized data
#     ax.plot(range(len(mean_standardized_neuron1)), mean_standardized_neuron2, mean_standardized_neuron1, color='black', linewidth=2)

    ax.set_xlabel('Time', fontsize = 14)
    ax.set_ylabel(f'Count of Neuron {neuron1_idx}', fontsize=14)
    ax.set_zlabel(f'Count of Neuron {neuron2_idx}', rotation=90, fontsize=14)
    ax.set_title(title, fontsize = 16)

#    ax.set_xlim(100, max(len(counting_process_nd_result[i][0]) for i in range(num_samples)))
#    ax.set_ylim(min(min(counting_process_nd_result[i][1] / theoretical_std_dev[1]) for i in range(num_samples)),
#                max(max(counting_process_nd_result[i][1] / theoretical_std_dev[1]) for i in range(num_samples)))
#    ax.set_zlim(min(min(counting_process_nd_result[i][0] / theoretical_std_dev[0]) for i in range(num_samples)),
#                max(max(counting_process_nd_result[i][0] / theoretical_std_dev[0]) for i in range(num_samples)))

    plt.show()


# In[14]:


def standardize_counts_loop(counts, theoretical_means, theoretical_std_devs, num_samples, num_neurons, exclude):
    """
    Loop-based standardization for N neurons.
    
    Parameters/Returns: Same as above.
    """
    # make note on the structure of input for later debugging.
    standardized_counts = np.zeros_like(counts[:,:,exclude:])  # Preserves input shape
    
    for i in range(num_samples):
        for n in range(num_neurons):  # Iterate over all neurons
            standardized_counts[i][n] = (counts[i][n][exclude:] - theoretical_means[n][exclude:]) / theoretical_std_devs[n][exclude:]
            # removes the very first entry so we don't divide by zero
    
    return standardized_counts


# In[15]:


def mean_trajectory(count):
    """Calculate mean trajectory for centered and standardized data"""
    mean_trajectory = np.mean(count, axis=0)
    std_trajectory = np.std(count, axis=0)
    
    # Center and standardize
    centered = mean_trajectory - np.mean(mean_trajectory)
    standardized = (mean_trajectory - np.mean(mean_trajectory)) / std_trajectory
    
    return centered, standardized


# In[16]:


from scipy import stats

def confidence_intervals(mean, std_dev, num_samples, confidence_level=0.95):
    """
    Calculate confidence intervals for neural data.
    
    Parameters:
    -----------
    mean : Mean values of your data. Could be mean spike counts/rates
    std_dev : Standard deviation of your data
    num_samples : Number of trials/samples
    confidence_level : Desired confidence level (default 0.95 for 95% confidence)
    
    Returns:
    --------
    lower_bound : Lower confidence bound
    upper_bound : Upper confidence bound
    """
    
    t_value = stats.t.ppf((1 + confidence_level) / 2, df=num_samples-1)
    
    # Calculate standard error
    standard_error = std_dev / np.sqrt(num_samples)
    
    # Calculate margin of error
    margin_of_error = t_value * standard_error
    
    # Calculate bounds
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return lower_bound.tolist(), upper_bound.tolist()


# In[17]:


def sim(N_E, epsilon, g, nu_ext_over_nu_thr, time, num_neurons, num_samples):
    """
    Parameters:
    -----------
    g : float
        Relative inhibitory to excitatory synaptic strength
    nu_ext_over_nu_thr : float
        Ratio of external stimulus rate to threshold rate
    sim_time : brian2.units.fundamentalunits.Quantity
        Simulation time in milliseconds
    ax_spikes : matplotlib.axes.Axes
        Axes to plot spikes on
    ax_rates : matplotlib.axes.Axes
        Axes to plot rates on
    rate_tick_step : float
        Step size for rate axis ticks
    """
    # # Network parameters
    # N_E = 1000
    gamma = 0.25
    N_I = round(gamma * N_E)
    N = N_E + N_I
    #epsilon = 0.1
    C_E = epsilon * N_E
    C_ext = C_E

    # Neuron parameters (all in Brian2 units)
    tau = 20 * ms
    theta = 20 * mV
    V_r = 10 * mV
    tau_rp = 2 * ms

    # # Synapse parameters
    J = 0.1 * mV
    D = 1.5 * ms

    # External stimulus
    nu_thr = theta / (J * C_E * tau)
    nu_ext = nu_ext_over_nu_thr * nu_thr

    # Set simulation timestep
    defaultclock.dt = 0.1 * ms

    # Create neuron groups
    neurons = NeuronGroup(N,
                         '''
                         dv/dt = -v/tau : volt (unless refractory)
                         ''',
                         threshold='v > theta',
                         reset='v = V_r',
                         refractory=tau_rp,
                         method='exact')

    # Split into excitatory and inhibitory populations
    excitatory_neurons = neurons[:N_E]
    inhibitory_neurons = neurons[N_E:]

    # Create synapses
    exc_synapses = Synapses(excitatory_neurons, neurons, 
                           on_pre='v_post += J',
                           delay=D)
    exc_synapses.connect(p=epsilon)

    inhib_synapses = Synapses(inhibitory_neurons, neurons, 
                             on_pre='v_post += -g*J',
                             delay=D)
    inhib_synapses.connect(p=epsilon)

    # Add external input
    external_poisson_input = PoissonInput(neurons, 'v', 
                                        N=C_ext,
                                        rate=nu_ext,
                                        weight=J)

    # Set up monitors
    rate_monitor_exc = PopulationRateMonitor(excitatory_neurons)
    rate_monitor_inh = PopulationRateMonitor(inhibitory_neurons)
    spike_monitor_exc = SpikeMonitor(excitatory_neurons[:num_neurons])
    spike_monitor_inh = SpikeMonitor(inhibitory_neurons[:num_neurons])

    # Create and run network
    net = Network(neurons, exc_synapses, inhib_synapses, 
                 external_poisson_input,
                 rate_monitor_exc, rate_monitor_inh,
                 spike_monitor_exc, spike_monitor_inh)

    #####################################
    excitatory_spikes = []
    inhibitory_spikes = []
    net.store()
    for sample in range(num_samples):
        net.restore()
        # reshuffles the intial voltages, unirand = net.v(shape)
        # need details like reset threshold (V_r), theta
        # after defining the net, extract back theta V_r, use theta and V_r to define a range and call a uniform random, define this as net.v.
        # once restored
        #### TESTING #####

        # Print PRE-reset voltages (should be identical across samples)
        #print(f"\nSample {sample} - PRE-RESET voltages (first 5 neurons):")
        #print(neurons.v[:5])  # First 5 neurons
    
        neurons.v = V_r + (theta - V_r) * np.random.rand(N)

        # Print POST-reset voltages (should differ across samples)
        #print(f"Sample {sample} - POST-RESET voltages (first 5 neurons):")
        #print(neurons.v[:5])

        #### END TESTING ####

        
        # Run the simulation
        net.run(time * ms, report=None)
        
        # Collect spike trains for this sample
        excitatory_spikes.append(spike_monitor_exc.spike_trains())
        inhibitory_spikes.append(spike_monitor_inh.spike_trains())
    
    #####################################

    # Get time ranges for plotting (in ms)
    # t_start = params["t_range"][0] * ms
    # t_end = params["t_range"][1] * ms

    # # Plot spikes
    # ax_spikes.plot(spike_monitor_exc.t/ms, 
    #                spike_monitor_exc.i,
    #                '|', color='blue', label='Excitatory')
    # ax_spikes.plot(spike_monitor_inh.t/ms,
    #                spike_monitor_inh.i + 25,
    #                '|', color='red', label='Inhibitory')

    # # Plot rates
    # ax_rates.plot(rate_monitor_exc.t/ms,
    #               rate_monitor_exc.rate/Hz,
    #               color='blue', label='Excitatory')
    # ax_rates.plot(rate_monitor_inh.t/ms,
    #               rate_monitor_inh.rate/Hz,
    #               color='red', label='Inhibitory')

    # # Configure plots
    # ax_spikes.set_yticks([])
    # ax_spikes.legend(loc='upper right')
    # ax_rates.legend(loc='upper right')
    
    # ax_spikes.set_xlim(t_start/ms, t_end/ms)
    # ax_rates.set_xlim(t_start/ms, t_end/ms)
    # ax_rates.set_ylim(*params["rate_range"])
    # ax_rates.set_xlabel("t [ms]")
    
    # ax_rates.set_yticks(np.arange(
    #     params["rate_range"][0],
    #     params["rate_range"][1] + rate_tick_step,
    #     rate_tick_step
    # ))

    # plt.subplots_adjust(hspace=0)

    return {
        'excitatory': {
            'spike_times': spike_monitor_exc.t,  # Keep Brian2 units
            'spike_indices': spike_monitor_exc.i,
            'rate_times': rate_monitor_exc.t,
            'rate_values': rate_monitor_exc.rate,
            'spike_trains': excitatory_spikes
        },
        'inhibitory': {
            'spike_times': spike_monitor_inh.t,  # Keep Brian2 units
            'spike_indices': spike_monitor_inh.i,
            'rate_times': rate_monitor_inh.t,
            'rate_values': rate_monitor_inh.rate,
            'spike_trains': inhibitory_spikes
        }
    }
# parameters = {
#     "C": {
#         "g": 7, # g is a good control for manipulating the firing rate
#         "nu_ext_over_nu_thr": 2,
#         "t_range": [1000, 1200],
#         "rate_range": [0, 200],
#         "rate_tick_step": 50,
#     },
# }

# for panel, params in parameters.items():
#     fig = plt.figure(figsize=(4, 5))
#     fig.suptitle(panel)

#     gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[4, 1])
#     ax_spikes, ax_rates = gs.subplots(sharex="col")

#     results = sim(
#         params["g"],
#         params["nu_ext_over_nu_thr"],
#         params["t_range"][1] * ms,
#         ax_spikes,
#         ax_rates,
#         params["rate_tick_step"],
#     )
    
#     # Print statistics for both populations
#     for pop_type in ['excitatory', 'inhibitory']:
#         print(f"\n{pop_type.capitalize()} population:")
#         print(f"Spike Times (first 10): {results[pop_type]['spike_times'][:10]}")
#         print(f"Spike Indices (first 10): {results[pop_type]['spike_indices'][:10]}")
#         print(f"Mean firing rate: {np.mean(results[pop_type]['rate_values'])} Hz")
#         print(f"Number of spikes: {len(results[pop_type]['spike_times'])}")

# plt.show()


# In[18]:


from brian2 import *
def create_connections(excitatory_data, inhibitory_data, p_connection=0.1, 
                      weight_exc=0.1*mV, weight_inh=-0.5*mV):
    """
    Create synaptic connections between neurons based on their spike train data.
    
    Parameters:
    -----------
    excitatory_data : dict
        Dictionary containing excitatory neuron data
    inhibitory_data : dict
        Dictionary containing inhibitory neuron data
    p_connection : float
        Connection probability (default 0.1)
    weight_exc : brian2.units.fundamentalunits.Quantity
        Weight for excitatory synapses
    weight_inh : brian2.units.fundamentalunits.Quantity
        Weight for inhibitory synapses
    
    Returns:
    --------
    tuple
        (exc_to_exc, exc_to_inh, inh_to_exc, inh_to_inh) Synapses objects
    """
    
    # Get number of neurons from spike trains
    N_exc = len(excitatory_data['spike_trains'])
    N_inh = len(inhibitory_data['spike_trains'])
    
    # Create neuron groups
    excitatory_neurons = NeuronGroup(N_exc,
                                   '''dv/dt = -v/tau : volt (unless refractory)
                                      tau : second''',
                                   threshold='v > 20*mV',
                                   reset='v = 0*mV',
                                   refractory=2*ms,
                                   method='exact')
    
    inhibitory_neurons = NeuronGroup(N_inh,
                                   '''dv/dt = -v/tau : volt (unless refractory)
                                      tau : second''',
                                   threshold='v > 20*mV',
                                   reset='v = 0*mV',
                                   refractory=2*ms,
                                   method='exact')
    
    # Create synapses with random connectivity and no self-connections
    
    # Excitatory to Excitatory
    exc_to_exc = Synapses(excitatory_neurons, excitatory_neurons,
                         model='w : volt',
                         on_pre='v_post += w')
    exc_to_exc.connect(condition='i != j', p=p_connection)  # No self-connections
    exc_to_exc.w = weight_exc
    
    # Excitatory to Inhibitory
    exc_to_inh = Synapses(excitatory_neurons, inhibitory_neurons,
                         model='w : volt',
                         on_pre='v_post += w')
    exc_to_inh.connect(p=p_connection)
    exc_to_inh.w = weight_exc
    
    # Inhibitory to Excitatory
    inh_to_exc = Synapses(inhibitory_neurons, excitatory_neurons,
                         model='w : volt',
                         on_pre='v_post += w')
    inh_to_exc.connect(p=p_connection)
    inh_to_exc.w = weight_inh
    
    # Inhibitory to Inhibitory
    inh_to_inh = Synapses(inhibitory_neurons, inhibitory_neurons,
                         model='w : volt',
                         on_pre='v_post += w')
    inh_to_inh.connect(condition='i != j', p=p_connection)  # No self-connections
    inh_to_inh.w = weight_inh
    
    # Print connection statistics
    print("\nConnection statistics:")
    print(f"E→E connections: {len(exc_to_exc.w)} "
          f"({len(exc_to_exc.w)/(N_exc**2)*100:.1f}% connected)")
    print(f"E→I connections: {len(exc_to_inh.w)} "
          f"({len(exc_to_inh.w)/(N_exc*N_inh)*100:.1f}% connected)")
    print(f"I→E connections: {len(inh_to_exc.w)} "
          f"({len(inh_to_exc.w)/(N_inh*N_exc)*100:.1f}% connected)")
    print(f"I→I connections: {len(inh_to_inh.w)} "
          f"({len(inh_to_inh.w)/(N_inh**2)*100:.1f}% connected)")
    
    return exc_to_exc, exc_to_inh, inh_to_exc, inh_to_inh


# In[19]:


def convert_spike_trains_to_binary(spike_trains_dict, time, num_neurons):
    """
    Convert Brian2 spike trains to binary arrays.
    
    Parameters:
    -----------
    spike_trains_dict : dict
        Dictionary of spike trains from Brian2 SpikeMonitor
    time : int
        Duration of simulation in ms
    num_neurons : int
        Number of neurons to convert
    
    Returns:
    --------
    spike_events : list
        List of binary arrays (1=spike, 0=no spike) for each neuron
    """
    spike_events = []
    
    for n in range(num_neurons):
        events = np.zeros(int(time))
        if n in spike_trains_dict:
            # Convert spike times to milliseconds and to indices
            spike_times = spike_trains_dict[n]
            spike_indices = (spike_times/ms).astype(int)
            # Only include spikes within the time window
            valid_indices = spike_indices[spike_indices < time]
            events[valid_indices] = 1
        spike_events.append(events)
    
    return spike_events


# In[20]:


import json

def save_spike_trains(spike_trains, filename):
    """
    Save spike trains to a JSON file.
    
    Parameters:
    -----------
    spike_trains : list of dict
        List of dictionaries containing spike trains
    filename : str
        Name of the file to save to (should end in .json)
    """
    serializable_spike_trains = []
    for sample in spike_trains:
        sample_dict = {}
        for neuron_id, times in sample.items():
            sample_dict[str(neuron_id)] = (times/second).tolist()
        serializable_spike_trains.append(sample_dict)
    
    with open(filename, 'w') as f:
        json.dump(serializable_spike_trains, f)
    
    print(f"Spike trains saved to {filename}")


# In[21]:


import json

def load_spike_trains(filename):
    """
    Load spike trains from a JSON file.
    
    Parameters:
    -----------
    filename : str
        Name of the file to load from
        
    Returns:
    --------
    spike_trains : list of dict
        List of dictionaries where each dictionary contains spike times for different neurons with Brian2 units
    """
    with open(filename, 'r') as f:
        loaded_spike_trains = json.load(f)
    
    spike_trains = []
    for sample in loaded_spike_trains:
        sample_dict = {}
        for neuron_id, times in sample.items():
            sample_dict[int(neuron_id)] = np.array(times) * second
        spike_trains.append(sample_dict)
    
    return spike_trains


# In[22]:


def standardize_units(spike_trains):
    """
    Converts all spike times in a spike trains dictionary to unitless values in seconds.

    Parameters:
    -----------
    spike_trains : list of dict
        List where each dictionary contains spike times for different neurons
        
    Returns:
    --------
    standardized_trains : list of dict
        List of dictionaries with all times converted to unitless values in seconds
    """
    standardized_trains = []
    
    for sample in spike_trains:
        converted_sample = {}
        for neuron_id, times in sample.items():
            converted_sample[neuron_id] = times/second
        standardized_trains.append(converted_sample)
    
    return standardized_trains


# In[23]:


def spikes_to_binary(spike_trains, num_neurons, time):
    """
    Convert spike times to binary sequences
    
    Parameters:
    -----------
    spike_trains : list of dictionaries
        Each dictionary contains spike times for neurons
    num_neurons : int
        Number of neurons in the data
    time : int
        Duration of recording in milliseconds
    time_resolution : float
        Time bin size in seconds (default=0.001 for millisecond resolution)
        
    Returns:
    --------
    binary_data : list of lists
        Each inner list contains numpy arrays (one per neuron) of 0s and 1s
    """
    binary_data = []
    
    for sample in spike_trains:
        sample_data = []
        
        # Process each neuron
        for neuron in range(num_neurons):
            # Create zero array with length equal to time parameter
            spike_array = np.zeros(time)
            
            # Get spike times for this neuron
            spike_times = sample[neuron]
            
            spike_indices = (spike_times * 1000).astype(int)
            
            # Set spikes to 1
            for idx in spike_indices:
                if 0 <= idx < time:  # Ensure index is within bounds
                    spike_array[idx] = 1
                    
            sample_data.append(spike_array)
            
        binary_data.append(sample_data)
    
    return binary_data


# In[24]:


def hotelling_t2_test(x, y=None, bessel=True, S=None):
    """
    Compute the Hotelling T² test statistic.
    
    Parameters:
    -----------
    x : array-like
        Samples of observations for one or two sample test (required)
    y : array-like, optional
        For two sample test: samples of observations
        For one sample test: list of means to test against
    bessel : bool, default=True
        Apply Bessel's correction to the covariance matrix
    S : array-like, optional
        Pre-computed covariance matrix for one-sample test
        
    Returns:
    --------
    dict
        Dictionary containing:
        - t2_stat: Hotelling's T² statistic
        - f_value: F statistic
        - p_value: p-value of the test
        - covariance: Covariance matrix used (pooled for two-sample test)
        - df1, df2: Degrees of freedom for F distribution
    """
    # Convert inputs to numpy arrays if they aren't already
    x = np.asarray(x)
    
    # Get dimensions
    try:
        nx, p = x.shape
    except ValueError:
        # Handle 1D array case
        nx = len(x)
        p = 1
        x = x.reshape(nx, p)
    
    # Calculate mean of x
    x_bar = np.mean(x, axis=0)
    
    # Determine if this is a one-sample or two-sample test
    one_sample = y is None or np.isscalar(y) or (isinstance(y, (list, np.ndarray)) and len(np.shape(y)) <= 1)
    
    if one_sample:
        # One-sample T² test
        if y is None:
            y = np.zeros(p)
        else:
            y = np.asarray(y)
            if len(y) != p:
                raise ValueError(f"Error: Mean vector must have same dimension as data ({len(y)} != {p}).")
        
        diff_bar = x_bar - y
        ny = None
        
        # Calculate covariance matrix
        if S is not None:
            cov = S
        else:
            if bessel:
                cov = np.cov(x, rowvar=False)
            else:
                cov = np.cov(x, rowvar=False, bias=True)
        
        # Calculate inverse of covariance matrix
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        
        # Calculate T² statistic
        t2_stat = nx * (diff_bar @ inv_cov @ diff_bar)
        
        # If S is provided, just return the T² statistic
        if S is not None:
            return {'t2_stat': t2_stat}
        
        # Calculate F statistic
        df1 = p
        df2 = nx - p
        f_value = df2 / (df1 * df1) * t2_stat
        
        # Calculate p-value
        p_value = stats.f.sf(f_value, df1, df2)
        
        return {
            't2_stat': t2_stat,
            'f_value': f_value,
            'p_value': p_value,
            'covariance': cov,
            'df1': df1,
            'df2': df2
        }
    
    else:
        # Two-sample T² test
        y = np.asarray(y)
        
        try:
            ny, py = y.shape
        except ValueError:
            # Handle 1D array case
            ny = len(y)
            py = 1
            y = y.reshape(ny, py)
        
        if p != py:
            raise ValueError(f"Error: The two samples must have the same number of features ({p} != {py}).")
        
        # Calculate mean of y
        y_bar = np.mean(y, axis=0)
        
        # Calculate difference of means
        diff_bar = x_bar - y_bar
        
        # Apply Bessel's correction if requested
        if bessel:
            n1 = nx - 1
            n2 = ny - 1
        else:
            n1 = nx
            n2 = ny
        
        n = n1 + n2
        
        # Calculate pooled covariance matrix
        if bessel:
            cov_x = np.cov(x, rowvar=False)
            cov_y = np.cov(y, rowvar=False)
            pooled_cov = ((nx - 1) * cov_x + (ny - 1) * cov_y) / (nx + ny - 2)
        else:
            cov_x = np.cov(x, rowvar=False, bias=True)
            cov_y = np.cov(y, rowvar=False, bias=True)
            pooled_cov = (nx * cov_x + ny * cov_y) / (nx + ny)
        
        # Calculate inverse of pooled covariance
        try:
            inv_pooled_cov = np.linalg.inv(pooled_cov)
        except np.linalg.LinAlgError:
            inv_pooled_cov = np.linalg.pinv(pooled_cov)
        
        # Calculate T² statistic
        t2_stat = (nx * ny) / (nx + ny) * (diff_bar @ inv_pooled_cov @ diff_bar)
        
        # Calculate F statistic
        df1 = p
        df2 = nx + ny - p - 1
        f_value = df2 / (df1 * (nx + ny - 2)) * t2_stat
        
        # Calculate p-value
        p_value = stats.f.sf(f_value, df1, df2)
        
        return {
            't2_stat': t2_stat,
            'f_value': f_value,
            'p_value': p_value,
            'covariance': pooled_cov,
            'df1': df1,
            'df2': df2
        }


# In[25]:


def serialize_counts(count_data):
    """Flatten 3D count data (samples × neurons × time) into 1D, treating neurons and time as replicates."""
    return count_data.ravel()  # Equivalent to np.con


# In[26]:


def covariance_diagonal(data, tol=1e-8):
    """
    Compute the covariance matrix of the input data and check if it is 
    a diagonal matrix with 1s on the diagonal (off-diagonals can be anything).

    Parameters:
        data (np.ndarray): Input data (shape: `n_samples × n_features`).
        tol (float): Numerical tolerance for checking 1s on the diagonal.

    Returns:
        tuple: (cov_matrix, is_diagonal_with_ones) 
            - cov_matrix: Computed covariance matrix.
            - is_diagonal_with_ones: True if diagonal is all 1s.
    """
    # Compute the covariance matrix (rowvar=False means columns are variables)
    cov_matrix = np.cov(data, rowvar=False)
    
    # Check if diagonal entries are 1 (within tolerance)
    diagonal_ones = np.allclose(np.diag(cov_matrix), 1.0, atol=tol)
    
    return cov_matrix, diagonal_ones


# In[27]:


def plot_covariance_matrix(cov_matrix, normalize=False, title=None, cmap='coolwarm', annot=True, neuron_labels=None):
    """
    Plot the covariance (or correlation) matrix of the input data using Matplotlib.

    Args:
        cov_matrix (np.ndarray): Covariance matrix to plot.
        normalize (bool): If True, normalizes to a correlation matrix (diagonal=1).
        title (str): Optional title for the plot.
        cmap (str): Colormap (e.g., 'coolwarm', 'viridis').
        annot (bool): If True, annotates cells with values.
        neuron_labels (list): Optional list of neuron labels (e.g., ['E0', 'E1', 'I0', 'I1']).
    Returns:
        np.ndarray: Computed covariance/correlation matrix.
    """
    
    # Normalize to correlation matrix if requested
    if normalize:
        std_dev = np.sqrt(np.diag(cov_matrix))
        cov_matrix = cov_matrix / np.outer(std_dev, std_dev)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    n_features = cov_matrix.shape[0]

    # Plot heatmap
    im = ax.imshow(cov_matrix, cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation' if normalize else 'Covariance', fontsize=12)

    # Annotate cells with values
    if annot:
        for i in range(n_features):
            for j in range(n_features):
                ax.text(j, i, f"{cov_matrix[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="black", fontsize=8)

    # Customize axes with neuron labels
    ticks = np.arange(n_features)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    if neuron_labels is not None:
        # Use provided neuron labels with color coding
        ax.set_xticklabels(neuron_labels, fontsize=10, rotation=45)
        ax.set_yticklabels(neuron_labels, fontsize=10)
        
        # Color-code the tick labels
        for i, label in enumerate(neuron_labels):
            if label.startswith('E'):
                color = 'red'  # Excitatory = red
            elif label.startswith('I'):
                color = 'blue'  # Inhibitory = blue
            else:
                color = 'black'  # Default
            
            # Color x-axis labels
            ax.get_xticklabels()[i].set_color(color)
            ax.get_xticklabels()[i].set_weight('bold')
            
            # Color y-axis labels
            ax.get_yticklabels()[i].set_color(color)
            ax.get_yticklabels()[i].set_weight('bold')
    else:
        # Default labeling
        ax.set_xticklabels([f"N {i+1}" for i in ticks], fontsize=12)
        ax.set_yticklabels([f"N {i+1}" for i in ticks], fontsize=12)

    # Title
    default_title = "Correlation Matrix" if normalize else "Covariance Matrix"
    ax.set_title(title if title else default_title, pad=20, fontsize=16)
    
    # Add legend for neuron types
    if neuron_labels is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Excitatory'),
            Patch(facecolor='blue', alpha=0.7, label='Inhibitory')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    return cov_matrix


# In[28]:


def combined_poisson_plot(spike_dict_list, counting_process_nd, theoretical_means, empirical_mean, colors=None, figsize=(12, 10)):
    """
    Creates a combined plot with:
    - Top: Count processes with theoretical and empirical means
    - Bottom: Raster plot of spike times
    
    Parameters:
    - spike_dict_list: List of dictionaries for raster plot {neuron_idx: spike_times}
    - counting_process_nd: Array of counting processes for each sample
    - theoretical_means: Theoretical mean values
    - empirical_mean: Empirical mean values
    - colors: Optional custom colors
    - figsize: Figure size
    """
    # Prepare figure with GridSpec for custom layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
    
    # Top axis for counts and means
    ax_top = plt.subplot(gs[0])
    
    # Bottom axis for raster plot
    ax_bottom = plt.subplot(gs[1], sharex=ax_top)
    
    # Get number of samples and neurons
    num_samples = len(spike_dict_list)
    max_neuron_idx = max(max(neuron_dict.keys()) for neuron_dict in spike_dict_list) + 1
    
    # Set default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max_neuron_idx))
    
    # --- Top Plot (Counts and Means) ---
    # Plot the counting processes
    for i in range(num_samples):
        ax_top.plot(counting_process_nd[i][0], color=colors[i % len(colors)], alpha=0.5, label=f'Sample {i}' if i < 5 else None)
    
    # Plot theoretical and empirical means
    ax_top.plot(theoretical_means[0], color='red', linewidth=2, label='Theoretical Mean')
    ax_top.plot(empirical_mean[0], color='blue', linewidth=2, linestyle='--', label='Empirical Mean')
    
    ax_top.set_title('Combined Poisson Processes Visualization')
    ax_top.set_ylabel('Count')
    ax_top.legend(loc='upper left')
    ax_top.grid(True, alpha=0.3)
    
    # --- Bottom Plot (Raster) ---
    # Prepare spike data
    spike_data = [[] for _ in range(max_neuron_idx)]
    for trial in spike_dict_list:
        for neuron_idx, spike_times in trial.items():
            spike_data[neuron_idx].extend(spike_times)
    
    # Create raster plot
    ax_bottom.eventplot(spike_data, colors=colors, linelengths=0.8)
    
    ax_bottom.set_title('Spike Raster Plot')
    ax_bottom.set_xlabel('Time')
    ax_bottom.set_ylabel('Neuron Index')
    ax_bottom.set_yticks(range(max_neuron_idx))
    ax_bottom.grid(True, alpha=0.3)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()


# In[29]:


def raster_plot(spike_dict_list, colors=None, linelengths=None, figsize=(12,8)):
    """
    Create a raster plot from a list of spike time dictionaries.
    
    Parameters:
    - spike_dict_list: List of dictionaries where each key is neuron index and value is spike times
    - colors: List of colors for each neuron (optional)
    - linelengths: List of line lengths for each neuron (optional)
    - figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Prepare data for eventplot
    max_neuron_idx = max(max(neuron_dict.keys()) for neuron_dict in spike_dict_list) + 1
    spike_data = [[] for _ in range(max_neuron_idx)]
    
    for trial in spike_dict_list:
        for neuron_idx, spike_times in trial.items():
            spike_data[neuron_idx].extend(spike_times)
    
    # Set default styling if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max_neuron_idx))
    if linelengths is None:
        linelengths = [0.5] * max_neuron_idx
    
    # Create the raster plot
    lines = plt.eventplot(spike_data, colors=colors, linelengths=linelengths)
    
    plt.title('Spike Raster Plot (Continuous)')
    plt.xlabel('Time')
    plt.ylabel('Neuron Index')
    plt.yticks(range(max_neuron_idx))
    plt.show()


# In[30]:


def plot_count_neuron1_vs_count_neuron2(counting_process_nd, num_samples):
    plt.figure(figsize=(8,8))  # Set the figure size to be a square
    for i in range(num_samples):
        plt.plot(counting_process_nd[i][0], counting_process_nd[i][1], label=f'Sample {i}')
        for j in range(len(counting_process_nd[i][0]) - 1):
            plt.plot([counting_process_nd[i][0][j], counting_process_nd[i][0][j+1]], [counting_process_nd[i][1][j], counting_process_nd[i][1][j]], color='#ff0000')
            plt.plot([counting_process_nd[i][0][j], counting_process_nd[i][0][j]], [counting_process_nd[i][1][j], counting_process_nd[i][1][j+1]], color='#0000ff')
    plt.xlabel('Count of Neuron 1')
    plt.ylabel('Count of Neuron 2')
    plt.title('Count of Neuron 1 vs Count of Neuron 2')
    max_val = max(max(counting_process_nd[i][0]) for i in range(num_samples))
    plt.xlim(0, max_val+4)  # Set x-axis upper limit to the maximum value
    plt.ylim(0, max_val+4)  # Set y-axis upper limit to the maximum value
    plt.gca().set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal
    plt.grid(True, linestyle='--', alpha=0.5)  # Add grid lines
    plt.show()


# In[31]:


def plot_count_neuron1_vs_count_neuron2_vs_count_neuron3(counting_process_nd, num_samples):
    fig = plt.figure(figsize=(8,8))  # Set the figure size to be a square
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    for i in range(num_samples):
        ax.plot(counting_process_nd[i][0], counting_process_nd[i][1], counting_process_nd[i][2], label=f'Sample {i}')
        for j in range(len(counting_process_nd[i][0]) - 1):
            ax.plot([counting_process_nd[i][0][j], counting_process_nd[i][0][j+1]], [counting_process_nd[i][1][j], counting_process_nd[i][1][j]], [counting_process_nd[i][2][j], counting_process_nd[i][2][j]], color='#ff0000')
            ax.plot([counting_process_nd[i][0][j], counting_process_nd[i][0][j]], [counting_process_nd[i][1][j], counting_process_nd[i][1][j]], [counting_process_nd[i][2][j], counting_process_nd[i][2][j+1]], color='#0000ff')
            ax.plot([counting_process_nd[i][0][j], counting_process_nd[i][0][j]], [counting_process_nd[i][1][j], counting_process_nd[i][1][j+1]], [counting_process_nd[i][2][j], counting_process_nd[i][2][j]], color='#00ff00')
    ax.set_xlabel('Count of Neuron 1')
    ax.set_ylabel('Count of Neuron 2')
    ax.set_zlabel('Count of Neuron 3')
    ax.set_title('Count of Neuron 1 vs Count of Neuron 2 vs Count of Neuron 3')
    max_val = max(max(counting_process_nd[i][0]) for i in range(num_samples))
    ax.set_xlim(0, max_val)  # Set x-axis upper limit to the maximum value
    ax.set_ylim(0, max_val)  # Set y-axis upper limit to the maximum value
    ax.set_zlim(0, max_val)  # Set z-axis upper limit to the maximum value
    plt.show()


# In[32]:


def find_nth_earliest_spike(spike_train, n):
    """
    Finds the nth earliest spike time across all neurons and samples.
    
    Args:
        spike_train (list of dict): List of samples, each with neuron:spike_times pairs.
        n (int): Which earliest spike to return (1=first, 2=second, etc.). Default=1.
        
    Returns:
        int or None: The nth earliest spike time (as integer). 
                    Returns None if there are fewer than n spikes.
    """
    all_spike_times = []
    for sample in spike_train:
        for spikes in sample.values():
            if len(spikes) > 0:
                all_spike_times.extend(spikes)
    
    if len(all_spike_times) >= n:  # Check if at least n spikes exist
        return int(np.sort(all_spike_times)[n-1])  # n-1 for 0-based indexing
    else:
        return None  # Not enough spikes


# In[33]:


def last_zero_std_time(empirical_std_dev, verbose=True):
    """
    For each neuron, finds the last timepoint where std=0 and returns safe exclude values.

    Args:
        empirical_std_dev: 2D array of shape (num_neurons, timepoints)
        verbose: If True, prints per-neuron details.

    Returns:
        exclude_times: List of exclude values (last_zero + 1) for each neuron
        global_exclude: Max exclude value across all neurons (safest)
    """
    exclude_times = []
    
    for n in range(empirical_std_dev.shape[0]):
        neuron_std = empirical_std_dev[n]
        zero_indices = np.where(neuron_std == 0)[0]
        
        last_zero_time = zero_indices[-1] if zero_indices.size > 0 else -1
        exclude_time = last_zero_time + 1
        exclude_times.append(exclude_time)
        
        if verbose:
            print(f"Neuron {n}: last zero std at t={last_zero_time}, recommended exclude={exclude_time}")
    
    global_exclude = max(exclude_times)
    
    if verbose:
        print(f"\nGlobal exclude (safest): t={global_exclude}")
    
    return exclude_times, global_exclude


# In[34]:


def plot_stats(theoretical_means, empirical_mean, theoretical_std_dev, empirical_std_dev, title = 'title'):
    """
    Plot the theoretical and empirical means with standard deviations.
    Includes debug prints and ideal reference lines.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find maximum length
    max_len = max(
        len(theoretical_means[0]) if theoretical_means.ndim > 1 else len(theoretical_means),
        len(empirical_mean[0]) if empirical_mean.ndim > 1 else len(empirical_mean),
        len(theoretical_std_dev[0]) if theoretical_std_dev.ndim > 1 else len(theoretical_std_dev),
        len(empirical_std_dev[0]) if empirical_std_dev.ndim > 1 else len(empirical_std_dev)
    )

    def pad_array(arr, max_length):
        if arr.ndim == 1:
            return np.concatenate([[0], arr, np.zeros(max(max_length - len(arr), 0))])
        else:
            padded = np.zeros((arr.shape[0], max_length + 1))
            for i in range(arr.shape[0]):
                filled_length = min(len(arr[i]), max_length)
                padded[i, :filled_length + 1] = np.concatenate([[0], arr[i][:filled_length]])
            return padded

    # Pad arrays - now properly handles 1D and 2D cases
    theory_mean_pad = pad_array(theoretical_means, max_len)
    theory_std_pad = pad_array(theoretical_std_dev, max_len)
    emp_mean_pad = pad_array(empirical_mean, max_len)
    emp_std_pad = pad_array(empirical_std_dev, max_len)

    # Extract first row if 2D (maintaining original behavior)
    if theory_mean_pad.ndim > 1:
        theory_mean_pad = theory_mean_pad[0]
        theory_std_pad = theory_std_pad[0]
    if emp_mean_pad.ndim > 1:
        emp_mean_pad = emp_mean_pad[0]
        emp_std_pad = emp_std_pad[0]

    time_points = np.arange(len(theory_mean_pad))

    # Plot theoretical and empirical data
    ax.plot(time_points, theory_mean_pad, label='Homogeneous Mean', color='#3498db', linewidth=2)
    ax.fill_between(time_points,
                  theory_mean_pad - theory_std_pad,
                  theory_mean_pad + theory_std_pad,
                  color='#3498db', alpha=0.3, label='Homogeneous Std Dev')
    
    ax.plot(time_points, emp_mean_pad, label='Inhomogeneous Mean', color='#e74c3c', linewidth=2)
    ax.fill_between(time_points,
                  emp_mean_pad - emp_std_pad,
                  emp_mean_pad + emp_std_pad,
                  color='#e74c3c', alpha=0.3, label='Inhomogeneous Std Dev')

    ax.axhline(y=-1, color='black', linestyle='-', linewidth=1.5)
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5)

    ax.set_title(title, fontsize = 16)
    ax.set_xlabel('Time', fontsize = 14)
    ax.set_ylabel('Value', fontsize = 14)
    
    # Auto-adjust y-axis limits if theoretical values are very small
    y_min = min(np.min(theory_mean_pad - theory_std_pad), np.min(emp_mean_pad - emp_std_pad), -1.1)
    y_max = max(np.max(theory_mean_pad + theory_std_pad), np.max(emp_mean_pad + emp_std_pad), 1.1)
    ax.set_ylim(y_min, y_max)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# In[35]:


def plot_multiple_stats(theoretical_means, empirical_mean, theoretical_std_dev, empirical_std_dev, title='title'):
    """
    Plot the theoretical and empirical means with standard deviations for multiple neurons.
    Creates separate subplots for each neuron.
    
    Parameters:
    -----------
    theoretical_means : array
        Theoretical mean values (2D array where each row is a neuron)
    empirical_mean : array  
        Empirical mean values (2D array where each row is a neuron)
    theoretical_std_dev : array
        Theoretical standard deviation values (2D array where each row is a neuron)
    empirical_std_dev : array
        Empirical standard deviation values (2D array where each row is a neuron)
    title : str, default='title'
        Base title for plots (will be appended with neuron number)
    """
    
    # Determine number of neurons from the data
    if theoretical_means.ndim > 1:
        num_neurons = theoretical_means.shape[0]
    elif empirical_mean.ndim > 1:
        num_neurons = empirical_mean.shape[0]
    elif theoretical_std_dev.ndim > 1:
        num_neurons = theoretical_std_dev.shape[0]
    elif empirical_std_dev.ndim > 1:
        num_neurons = empirical_std_dev.shape[0]
    else:
        print("Warning: All inputs are 1D. Cannot create multiple neuron plots.")
        return
    
    # Calculate subplot layout
    cols = min(3, num_neurons)  # Max 3 columns for readability
    rows = (num_neurons + cols - 1) // cols  # Ceiling division
    
    # Increased figure size and added extra spacing
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    # Handle different subplot configurations
    if num_neurons == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    def pad_array(arr, max_length):
        if arr.ndim == 1:
            return np.concatenate([[0], arr, np.zeros(max(max_length - len(arr), 0))])
        else:
            padded = np.zeros((arr.shape[0], max_length + 1))
            for i in range(arr.shape[0]):
                filled_length = min(len(arr[i]), max_length)
                padded[i, :filled_length + 1] = np.concatenate([[0], arr[i][:filled_length]])
            return padded
    
    # Create a plot for each neuron
    for neuron in range(num_neurons):
        ax = axes[neuron]
        
        # Extract data for this neuron
        theory_mean_neuron = theoretical_means[neuron] if theoretical_means.ndim > 1 else theoretical_means
        theory_std_neuron = theoretical_std_dev[neuron] if theoretical_std_dev.ndim > 1 else theoretical_std_dev
        emp_mean_neuron = empirical_mean[neuron] if empirical_mean.ndim > 1 else empirical_mean
        emp_std_neuron = empirical_std_dev[neuron] if empirical_std_dev.ndim > 1 else empirical_std_dev
        
        # Find maximum length for this neuron
        max_len = max(
            len(theory_mean_neuron),
            len(emp_mean_neuron),
            len(theory_std_neuron),
            len(emp_std_neuron)
        )
        
        # Pad arrays for this neuron
        theory_mean_pad = pad_array(theory_mean_neuron, max_len)
        theory_std_pad = pad_array(theory_std_neuron, max_len)
        emp_mean_pad = pad_array(emp_mean_neuron, max_len)
        emp_std_pad = pad_array(emp_std_neuron, max_len)
        
        # Handle case where pad_array still returns 2D (shouldn't happen for individual neuron data)
        if theory_mean_pad.ndim > 1:
            theory_mean_pad = theory_mean_pad[0]
            theory_std_pad = theory_std_pad[0]
        if emp_mean_pad.ndim > 1:
            emp_mean_pad = emp_mean_pad[0]
            emp_std_pad = emp_std_pad[0]
        
        time_points = np.arange(len(theory_mean_pad))
        
        # Plot theoretical and empirical data for this neuron
        ax.plot(time_points, theory_mean_pad, label='Homogeneous Mean', color='#3498db', linewidth=2)
        ax.fill_between(time_points,
                      theory_mean_pad - theory_std_pad,
                      theory_mean_pad + theory_std_pad,
                      color='#3498db', alpha=0.3, label='Homogeneous Std Dev')
        
        ax.plot(time_points, emp_mean_pad, label='Inhomogeneous Mean', color='#e74c3c', linewidth=2)
        ax.fill_between(time_points,
                      emp_mean_pad - emp_std_pad,
                      emp_mean_pad + emp_std_pad,
                      color='#e74c3c', alpha=0.3, label='Inhomogeneous Std Dev')

        ax.axhline(y=-1, color='black', linestyle='-', linewidth=1.5)
        ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5)

    
        # Reduced title font size to prevent overlap
        ax.set_title(f'{title} - Neuron {neuron}', fontsize=12)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        
        # Auto-adjust y-axis limits
        y_min = min(np.min(theory_mean_pad - theory_std_pad), np.min(emp_mean_pad - emp_std_pad), -1.1)
        y_max = max(np.max(theory_mean_pad + theory_std_pad), np.max(emp_mean_pad + emp_std_pad), 1.1)
        ax.set_ylim(y_min, y_max)
        
        ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots if any
    for i in range(num_neurons, len(axes)):
        axes[i].set_visible(False)
    
    # Enhanced spacing adjustments
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    plt.show()


# In[36]:


def data_plot(stands, num_samples, theoretical_means, empirical_mean, 
              theoretical_std_dev, empirical_std_dev, num_neurons, title=None,
              figsize=(12, 7), max_samples_to_show=5):
    """
    Plot standardized statistics and stands data with simple, clear lines.
    
    Parameters:
    -----------
    stands : array-like
        Standardized counts with shape (num_samples, num_neurons, time_points)
    num_samples : int
        Number of samples
    theoretical_means : array-like
        Theoretical mean values with shape (num_neurons, time_points)
    empirical_mean : array-like
        Empirical mean values with shape (num_neurons, time_points)
    theoretical_std_dev : array-like
        Theoretical standard deviation values with shape (num_neurons, time_points)
    empirical_std_dev : array-like
        Empirical standard deviation values with shape (num_neurons, time_points)
    num_neurons : int
        Number of neurons
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height) in inches
    max_samples_to_show : int, optional
        Maximum number of sample traces to show
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Create second y-axis for stands data
    ax2 = ax1.twinx()
    
    # Define vibrant colors for stands data
    stands_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF5', '#F5FF33']
    # Define lighter colors for statistics
    stat_colors = ['#0066CC', '#CC0000']  # Blue for theoretical, Red for empirical
    
    # Limit samples to show
    samples_to_show = min(num_samples, max_samples_to_show)
    
    # Store handles for legend
    legend_handles_ax1 = []  # For statistics (left axis)
    legend_handles_ax2 = []  # For standardized counts (right axis)
    
    # Plot statistics for each neuron
    for neuron in range(num_neurons):
        # Plot theoretical and empirical statistics on left axis
        theo_mean_line = ax1.plot(theoretical_means[neuron], linestyle='-', linewidth=2, 
                                 color=stat_colors[0], alpha=0.7)[0]
        theo_std_line = ax1.plot(theoretical_std_dev[neuron], linestyle='--', linewidth=2, 
                                color=stat_colors[0], alpha=0.5)[0]
        emp_mean_line = ax1.plot(empirical_mean[neuron], linestyle='-', linewidth=2, 
                                color=stat_colors[1], alpha=0.7)[0]
        emp_std_line = ax1.plot(empirical_std_dev[neuron], linestyle='--', linewidth=2, 
                               color=stat_colors[1], alpha=0.5)[0]
        
        # Add to legend handles (only for first neuron to avoid duplicates)
        if neuron == 0:
            legend_handles_ax1.extend([
                theo_mean_line, theo_std_line, emp_mean_line, emp_std_line
            ])
        
        # Plot stands data on right axis with bold, solid lines
        for i in range(samples_to_show):
            neuron_color = stands_colors[neuron % len(stands_colors)]
            count_line = ax2.plot(stands[i][neuron], linewidth=1, alpha=0.8,
                                 color=neuron_color)[0]
            
            # Add to legend handles (only first sample per neuron)
            if i == 0:
                legend_handles_ax2.append(count_line)
    
    # Set the title and labels
    if title is None:
        title = 'Neuron Statistics and Standardized Counts'
    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Statistical Values', fontsize=14)
    ax2.set_ylabel('Standardized Count', fontsize=14, color=stands_colors[0])
    
    # Set y-axis limits
    # For statistics data (left axis)
    y_min = min(np.min(theoretical_means), np.min(empirical_mean), 
               np.min(theoretical_std_dev), np.min(empirical_std_dev)) - 1
    y_max = max(np.max(theoretical_means), np.max(empirical_mean),
               np.max(theoretical_std_dev), np.max(empirical_std_dev)) + 1
    ax1.set_ylim(y_min, y_max)
    
    # For stands data (right axis)
    stands_min = float('inf')
    stands_max = float('-inf')
    for i in range(samples_to_show):
        for neuron in range(num_neurons):
            stands_min = min(stands_min, np.min(stands[i][neuron]))
            stands_max = max(stands_max, np.max(stands[i][neuron]))
    
    # Add margin to both min and max
    y_margin = (stands_max - stands_min) * 0.1
    ax2.set_ylim(stands_min - y_margin, stands_max + y_margin)
    
    # Create custom legend labels
    legend_labels_ax1 = [
        'Homogeneous Mean', 'Homogeneous Std Dev', 
        'Inhomogeneous', 'Inhomogeneous Std Dev'
    ]
    
    legend_labels_ax2 = [f'Neuron {n} Count' for n in range(num_neurons)]
    
    # Combine legend handles and labels
    all_handles = legend_handles_ax1 + legend_handles_ax2
    all_labels = legend_labels_ax1 + legend_labels_ax2
    
    # Add a combined legend
    ax1.legend(all_handles, all_labels, loc='upper left', fontsize=10, 
              ncol=2 if num_neurons > 2 else 1)
    
    # Add grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig, ax1, ax2


# In[37]:


#### TESTING ######


# In[38]:


def standardize_per_neuron(data_3d, exclude=0, eps=1e-8, verbose=True):
    """
    Standardizes a 3D array (samples, neurons, time) per neuron, excluding early unstable timepoints.

    Args:
        data_3d: Input array of shape (num_samples, num_neurons, num_timepoints).
        exclude: Number of initial timepoints to exclude (default: 0).
        eps: Small value to prevent division by zero (default: 1e-8).
        verbose: If True, prints post-standardization checks (default: True).

    Returns:
        Standardized array of same shape as input.
    """
    standardized_data = data_3d.copy()  # Preserve original

    for neuron in range(standardized_data.shape[1]):
        # Extract data (samples × time) excluding unstable points
        neuron_data = standardized_data[:, neuron, exclude:]

        # Compute mean and std (across samples and remaining timepoints)
        mu = np.mean(neuron_data)
        sigma = np.std(neuron_data)

        # Standardize (add epsilon to avoid division by zero)
        if sigma == 0:
            if verbose:
                print(f"Warning: Neuron {neuron} has std=0 after exclude={exclude}. Using eps={eps}.")
            sigma = eps

        standardized_data[:, neuron, exclude:] = (neuron_data - mu) / sigma

    if verbose:
        # Verify mean≈0 and std≈1 for each neuron (post-exclusion)
        means = np.mean(standardized_data[:, :, exclude:], axis=(0, 2))
        stds = np.std(standardized_data[:, :, exclude:], axis=(0, 2))

        print("\n=== Post-standardization checks (excluded t <", exclude, ") ===")
        print(f"Max |mean|: {np.max(np.abs(means)):.3e} (should be close to 0)")
        print(f"Max |std - 1|: {np.max(np.abs(stds - 1)):.3e} (should be close to 0)")

    return standardized_data


# In[39]:


def cov_standardized_data(X):
    """Compute covariance for standardized data (neurons × timepoints)."""
    # Since mean = 0, just compute (1/(T-1)) * (X @ X.T)
    return X @ X.T / (X.shape[1] - 1)


# In[40]:


def plot_covariance_manual(data):
    """
    Compute and plot the covariance matrix for standardized data.
    - Assumes data is already standardized (mean=0, std=1).
    - Input shape: (num_observations, num_features) or (num_features, num_observations).
    """
    # Ensure correct orientation (rows=samples, cols=features)
    if data.shape[0] > data.shape[1]:  # If more rows than columns, likely correct
        pass  # Assume shape is (observations, features)
    else:
        data = data.T  # Transpose to (observations, features)

    # Compute covariance (for standardized data)
    n = data.shape[0]  # Number of observations
    cov_matrix = (data.T @ data) / (n - 1)  # Same as np.cov(data.T)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cov_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    # Add text annotations (values)
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            val = cov_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", 
                    ha="center", va="center", 
                    color="black" if abs(val) < 0.5 else "white")

    # Customize plot
    num_neurons = cov_matrix.shape[0]
    ax.set_xticks(np.arange(num_neurons))
    ax.set_yticks(np.arange(num_neurons))
    ax.set_xticklabels([f"Neuron {i}" for i in range(num_neurons)])
    ax.set_yticklabels([f"Neuron {j}" for j in range(num_neurons)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Covariance", rotation=-90, va="bottom")

    ax.set_title("Neuron Covariance Matrix (Standardized Data)")
    plt.tight_layout()
    plt.show()


# In[41]:


def check_mean_with_se(data, expected_mean=None, confidence=0.95):
    """
    Calculate mean ± standard error (SE) and evaluate if an expected mean falls within the confidence interval (CI).

    Parameters:
    - data: Array-like input data.
    - expected_mean: Mean value to test (optional).
    - confidence: Confidence level for CI (default: 0.95 for 95%).

    Returns:
    - Dictionary with mean, SE, CI, interpretation, and (if expected_mean provided) a check result.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaN/Inf
    if len(data) == 0:
        raise ValueError("Input data contains no valid numbers after NaN removal.")

    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error (handles ddof=1)
    
    # Handle edge case: all values identical
    if np.isnan(se) or np.allclose(data, data[0]):
        se = 0.0
        ci_low = ci_high = mean
        interpretation = "All data points are identical (SE=0)."
    else:
        ci_low, ci_high = stats.t.interval(confidence, df=n-1, loc=mean, scale=se)
        interpretation = (
            f"The true population mean is estimated to be between {ci_low:.2f} and {ci_high:.2f} "
            f"(with {int(confidence*100)}% confidence)."
        )
    
    result = {
        'mean': mean,
        'se': se,
        f'CI_{int(confidence*100)}%': (ci_low, ci_high),
        'interpretation': interpretation,
    }
    
    if expected_mean is not None:
        is_in_ci = (ci_low <= expected_mean <= ci_high)
        result['is_expected_in_CI'] = is_in_ci
        result['expected_mean_check'] = (
            f"The expected mean ({expected_mean}) is {'WITHIN' if is_in_ci else 'OUTSIDE'} "
            f"the {int(confidence*100)}% confidence interval."
        )
    
    return result


# In[42]:


def analyze_poisson_goodness_of_fit(standardized_data, plot_title="Poisson Goodness of Fit", sample_limit=5000):
    """
    Analyze goodness of fit specifically for standardized Poisson data.
    
    Parameters:
    -----------
    standardized_data : numpy.ndarray
        Flattened array of standardized data
    plot_title : str
        Title for the plot
    sample_limit : int
        Maximum number of data points to use for visualization
        
    Returns:
    --------
    dict
        Dictionary containing test results and statistics
    """
    
    # Calculate basic statistics
    mean = np.mean(standardized_data)
    std = np.std(standardized_data)
    
    # Perform Kolmogorov-Smirnov test (most appropriate for this case)
    ks_result = stats.kstest(standardized_data, 'norm', args=(0, 1))
    
    # Create figure with two key plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Q-Q plot (most informative for normality assessment)
    osm, osr = stats.probplot(standardized_data, dist="norm", fit=False)
    ax1.plot(osm, osr, 'bo', label='Data Quantiles')
    ax1.plot(osm, osm, 'r-', label='Standard Normal (μ=0, σ=1)')  # Force y=x line for N(0,1)
    # stats.probplot(standardized_data, dist="norm", plot=ax1)
    ax1.set_title('Q-Q Plot', fontsize = 16)
    ax1.set_xlabel('Theoretical Quantiles N(0,1)', fontsize=14)
    ax1.set_ylabel('Sample Quantiles', fontsize=14)
    
    # 2. Histogram with normal curve overlay
    counts, bins, _ = ax2.hist(standardized_data, bins=30, density=True, alpha=0.7)
    x = np.linspace(min(standardized_data), max(standardized_data), 1000)
    ax2.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, 
             label=f'Standard Normal')
    ax2.plot(x, stats.norm.pdf(x, mean, std), 'g--', lw=2, 
             label=f'Fitted: μ={mean:.3f}, σ={std:.3f}')
    
    # Add axis labels to histogram
    ax2.set_xlabel('Standardized Value', fontsize = 14)
    ax2.set_ylabel('Probability Density', fontsize = 14)
    
    ax2.set_title('Histogram with Normal Curve', fontsize = 16)
    ax2.legend()
    
    # Add test statistics as text
    textstr = '\n'.join([
        f"Mean: {mean:.4f} (Expected: 0)",
        f"Std Dev: {std:.4f} (Expected: 1)",
        f"KS Test p-value: {ks_result.pvalue:.6f}",
        f"KS Test statistic: {ks_result.statistic:.6f}"
    ])
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Set overall title
    fig.suptitle(plot_title, fontsize=14)
    plt.tight_layout()
    
    # Prepare results dictionary
    results = {
        'mean': mean,
        'std': std,
        'kolmogorov_smirnov': {
            'statistic': ks_result.statistic,
            'pvalue': ks_result.pvalue,
            'is_normal': ks_result.pvalue > 0.05
        }
    }
    
    return results, fig


# In[43]:


def center_counts_loop(counts, theoretical_means, num_samples, num_neurons):
    """
    Loop-based standardization for N neurons.
    
    Parameters/Returns: Same as above.
    """
    # make note on the structure of input for later debugging.
    centered_counts = np.zeros_like(counts[:,:,:])  # Preserves input shape
    
    for i in range(num_samples):
        for n in range(num_neurons):  # Iterate over all neurons
            centered_counts[i][n] = (counts[i][n][:] - theoretical_means[n][:])
            # removes the very first entry so we don't divide by zero
    
    return centered_counts


# In[44]:


def create_balanced_ei_dataset(excitatory_data, inhibitory_data, n_neurons_each=5):
    """
    Create a balanced dataset with equal numbers of excitatory and inhibitory neurons.
    
    Parameters:
    -----------
    excitatory_data : np.ndarray
        Shape (n_neurons, n_samples_x_time) - excitatory neuron data
    inhibitory_data : np.ndarray  
        Shape (n_neurons, n_samples_x_time) - inhibitory neuron data
    n_neurons_each : int
        Number of neurons to take from each type
    
    Returns:
    --------
    balanced_data : np.ndarray
        Shape (2*n_neurons_each, n_samples_x_time) - combined data
    neuron_labels : list
        Labels indicating neuron type and original index
    neuron_types : list
        Simple list of 'E' or 'I' for each neuron
    """
    exc_time_length = excitatory_data.shape[1]
    inh_time_length = inhibitory_data.shape[1]
    min_time_length = min(exc_time_length, inh_time_length)
    
    # Take first n_neurons_each from each dataset
    exc_subset = excitatory_data[:n_neurons_each, :min_time_length]  # Shape: (5, 99800)
    inh_subset = inhibitory_data[:n_neurons_each, :min_time_length]  # Shape: (5, 99800)
    
    # Concatenate along neuron axis (axis=0)
    balanced_data = np.concatenate([exc_subset, inh_subset], axis=0)  # Shape: (10, 99900)
    
    # Create labels
    neuron_labels = [f'E{i}' for i in range(n_neurons_each)] + [f'I{i}' for i in range(n_neurons_each)]
    neuron_types = ['E'] * n_neurons_each + ['I'] * n_neurons_each
    
    return balanced_data, neuron_labels, neuron_types


# In[45]:


def create_interleaved_ei_dataset(excitatory_data, inhibitory_data, n_neurons_each=5):
    """
    Create a balanced dataset with alternating excitatory and inhibitory neurons.
    Pattern: E0, I0, E1, I1, E2, I2, ...
    """
    exc_time_length = excitatory_data.shape[1]
    inh_time_length = inhibitory_data.shape[1]
    min_time_length = min(exc_time_length, inh_time_length)
    
    exc_subset = excitatory_data[:n_neurons_each, :min_time_length]  # Shape: (5, 99800)
    inh_subset = inhibitory_data[:n_neurons_each, :min_time_length]  # Shape: (5, 99800)
    
    # Create interleaved pattern
    balanced_data = np.zeros((2 * n_neurons_each, excitatory_data.shape[1]))
    neuron_labels = []
    neuron_types = []
    
    for i in range(n_neurons_each):
        # Place excitatory neuron
        balanced_data[2*i, :] = exc_subset[i, :]
        neuron_labels.append(f'E{i}')
        neuron_types.append('E')
        
        # Place inhibitory neuron  
        balanced_data[2*i + 1, :] = inh_subset[i, :]
        neuron_labels.append(f'I{i}')
        neuron_types.append('I')
    
    return balanced_data, neuron_labels, neuron_types


# In[46]:


def create_random_ei_dataset(excitatory_data, inhibitory_data, n_neurons_each=5, random_seed=42):
    """
    Create a balanced dataset with randomly selected and arranged neurons.
    """
    np.random.seed(random_seed)
    
    # Randomly select neurons from each type
    exc_indices = np.random.choice(excitatory_data.shape[0], n_neurons_each, replace=False)
    inh_indices = np.random.choice(inhibitory_data.shape[0], n_neurons_each, replace=False)

    exc_time_length = excitatory_data.shape[1]
    inh_time_length = inhibitory_data.shape[1]
    min_time_length = min(exc_time_length, inh_time_length)
    
    exc_subset = excitatory_data[exc_indices, :min_time_length]
    inh_subset = inhibitory_data[inh_indices, :min_time_length]
    
    # Combine and shuffle
    combined_data = np.concatenate([exc_subset, inh_subset], axis=0)
    combined_labels = [f'E{i}' for i in exc_indices] + [f'I{i}' for i in inh_indices]
    combined_types = ['E'] * n_neurons_each + ['I'] * n_neurons_each
    
    # Shuffle the order
    shuffle_indices = np.random.permutation(2 * n_neurons_each)
    balanced_data = combined_data[shuffle_indices, :]
    neuron_labels = [combined_labels[i] for i in shuffle_indices]
    neuron_types = [combined_types[i] for i in shuffle_indices]
    
    return balanced_data, neuron_labels, neuron_types


# In[47]:


from scipy.special import gammaln

def log_likelihood(observed_counts, predicted_counts):
    """
    Calculate log-likelihood for Poisson process.
    
    Parameters:
    -----------
    observed_counts : np.ndarray
        Observed spike counts (shape: n_samples x n_neurons x time)
    predicted_rates : np.ndarray  
        Predicted rates from theoretical model (same shape as observed_counts)
    
    Returns:
    --------
    log_likelihood : float
        Total log-likelihood
    log_likelihood_per_sample : np.ndarray
        Log-likelihood for each sample
    """
    # Poisson log-likelihood: k*log(λ) - λ - log(k!)
    # Using gammaln(k+1) instead of log(k!) for numerical stability
    
    log_likelihood_matrix = (observed_counts * np.log(predicted_counts + 1e-10) - 
                           predicted_counts - 
                           gammaln(observed_counts + 1))
    
    # Handle different dimensions
    if observed_counts.ndim == 1:
        # 1D array - single sample
        log_likelihood_per_sample = np.sum(log_likelihood_matrix)
        total_log_likelihood = log_likelihood_per_sample
        
    elif observed_counts.ndim == 2:
        # 2D array - num_samples, (neurons x time)
        # Assume first dimension is samples, sum over features
        log_likelihood_per_sample = np.sum(log_likelihood_matrix, axis=1)
        total_log_likelihood = np.sum(log_likelihood_per_sample)
        
    elif observed_counts.ndim == 3:
        # 3D array - (samples, neurons, time)
        # Sum over neurons and time for each sample
        log_likelihood_per_sample = np.sum(log_likelihood_matrix, axis=(1, 2))
        total_log_likelihood = np.sum(log_likelihood_per_sample)
        
    else:
        raise ValueError(f"Unsupported array dimension: {observed_counts.ndim}D")
    
    return total_log_likelihood, log_likelihood_per_sample


# In[48]:


from scipy.stats import multivariate_normal
import numpy as np

def multivariate_log_likelihood(data):
    """
    Compute multivariate normal log-likelihood for neural data.
    Treats each timepoint as an observation of the neuron population.
    
    Parameters:
    -----------
    data : array, shape (n_neurons, n_timepoints)
        Standardized neural data
    
    Returns:
    --------
    log_likelihood : float
        Total log-likelihood (sum across all timepoints)
    log_likelihood_per_sample : np.ndarray
        Log-likelihood for each timepoint/sample
    """
    
    # Reshape: each timepoint is a sample, each neuron is a feature
    data_reshaped = data.T  # Shape: (n_timepoints, n_neurons)
    
    # For standardized data, mean should be zero
    mean_vec = np.zeros(data.shape[0])  # Length: n_neurons
    
    # Covariance between neurons
    cov_matrix = cov_standardized_data(data)  # Shape: (n_neurons, n_neurons)
    
    # Add small regularization to avoid singular matrix
    cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
    
    # Compute log-likelihood for each timepoint (sample)
    log_likelihood_per_sample = multivariate_normal.logpdf(data_reshaped, mean=mean_vec, cov=cov_matrix)
    
    # Total log-likelihood
    total_log_likelihood = np.sum(log_likelihood_per_sample)
    
    return total_log_likelihood, log_likelihood_per_sample


# In[49]:


def proper_chi_square_test(observed, expected, n_bins=50, min_expected=5):
    """
    Proper chi-square test by binning continuous data with safety checks
    """
    # Flatten data
    obs_flat = observed.flatten()
    exp_flat = expected.flatten()
    
    # get ranges
    min_val = min(np.min(obs_flat), np.min(exp_flat))
    max_val = max(np.max(obs_flat), np.max(exp_flat))

    # from [min_val, ... , max_val] breaks up into n_bins+1 segments
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    # how many values in each bin
    obs_counts, _ = np.histogram(obs_flat, bins=bins)
    exp_counts, _ = np.histogram(exp_flat, bins=bins)
    
    # Remove bins where expected count is too small (| means or)
    mask = (obs_counts > 0) | (exp_counts > 0)
    mask = mask & (exp_counts >= min_expected)
    
    obs_clean = obs_counts[mask]
    exp_clean = exp_counts[mask]
    
    # Check if we have enough bins left
    if len(obs_clean) < 2:
        return {
            'error': 'Too few bins remaining after filtering',
            'bins_remaining': len(obs_clean),
            'bins_requested': n_bins,
            'bins_created': len(obs_counts)
        }
    
    # Perform chi-square test
    chi2_stat, p_value = stats.chisquare(obs_clean, exp_clean, sum_check=False)
    
    return {
        'chi_square_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': len(obs_clean) - 1,
        'bins_created': len(obs_counts),
        'bins_used': len(obs_clean),
        'bins_removed': len(obs_counts) - len(obs_clean)
    }


# In[50]:


def t_test(expected, observed, test_type='two_sample', target_mean=None):
    """T-test using SciPy for better numerical stability"""
    
    if test_type == 'one_sample':
        
        if target_mean is not None:
            t_stat, p_value = stats.ttest_1samp(expected.flatten(), target_mean)
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'test_type': 'one_sample_vs_mean',
                'target_mean': float(target_mean),
                'sample_mean': float(np.mean(expected.flatten()))
            }
        else:
            # Original behavior: test against 0
            t_stat, p_value = stats.ttest_1samp(expected.flatten(), 0)
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'test_type': 'one_sample_vs_zero'
            }
        
    elif test_type == 'two_sample':
        t_stat, p_value = stats.ttest_ind(expected.flatten(), observed.flatten())
        
    elif test_type == 'paired':
        # Handle shape matching
        data1_flat = expected.flatten()
        data2_flat = observed.flatten()
        min_size = min(len(data1_flat), len(data2_flat))
        
        t_stat, p_value = stats.ttest_rel(data1_flat[:min_size], data2_flat[:min_size])
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'test_type': test_type
    }


# In[51]:


def match_array_shapes(array1, array2):
    """
    Make two arrays the same shape by trimming or padding
    
    Args:
        array1: First numpy array (e.g., shape (10, 9990))
        array2: Second numpy array (e.g., shape (10, 9900))
        method: 'trim' (cut to smaller) or 'pad' (extend to larger)
    
    Returns:
        tuple: (array1_resized, array2_resized) with matching shapes
    """
    
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    
    print(f"Original shapes: {array1.shape} and {array2.shape}")
    
    # Get dimensions
    rows1, cols1 = array1.shape
    rows2, cols2 = array2.shape
    
    # Handle row mismatch (if needed)
    min_rows = min(rows1, rows2)
    array1 = array1[:min_rows, :]
    array2 = array2[:min_rows, :]
    
    # Handle column mismatch
    if cols1 != cols2:
    # Trim to smaller number of columns
        min_cols = min(cols1, cols2)
        array1_resized = array1[:, :min_cols]
        array2_resized = array2[:, :min_cols]
        print(f"Trimmed both to shape: {array1_resized.shape}")
    else:
        array1_resized = array1
        array2_resized = array2
    
    print(f"Final shapes: {array1_resized.shape} and {array2_resized.shape}")
    return array1_resized, array2_resized


# In[52]:


def check_standardization(mean, std_dev, tolerance=0.01):
    """
    Check how far mean and standard deviation are from ideal standardized values.
    
    Parameters:
    -----------
    mean : float or array-like
        Mean value(s) to check
    std_dev : float or array-like
        Standard deviation value(s) to check
    tolerance : float, default=0.01
        Acceptable tolerance for "close enough" (optional)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'mean_distance': Distance from 0
        - 'std_distance': Distance from 1
        - 'mean_ok': Boolean if within tolerance
        - 'std_ok': Boolean if within tolerance
        - 'both_ok': Boolean if both within tolerance
    """
    # Calculate distances
    mean_distance = np.abs(mean - 0)
    std_distance = np.abs(std_dev - 1)
    
    # Check if within tolerance
    mean_ok = mean_distance <= tolerance
    std_ok = std_distance <= tolerance
    both_ok = mean_ok and std_ok
    
    # Print results
    print("=" * 60)
    print("STANDARDIZATION CHECK")
    print("=" * 60)
    print(f"Mean:")
    print(f"  Actual value:    {mean:.6f}")
    print(f"  Target value:    0.000000")
    print(f"  Distance from 0: {mean_distance:.6f}")
    print(f"  Status:          {'✓ PASS' if mean_ok else '✗ FAIL'} (tolerance: {tolerance})")
    print()
    print(f"Standard Deviation:")
    print(f"  Actual value:    {std_dev:.6f}")
    print(f"  Target value:    1.000000")
    print(f"  Distance from 1: {std_distance:.6f}")
    print(f"  Status:          {'✓ PASS' if std_ok else '✗ FAIL'} (tolerance: {tolerance})")
    print()
    print(f"Overall: {'✓ BOTH PASS' if both_ok else '✗ FAILED'}")
    print("=" * 60)
    
    return {
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'mean_ok': mean_ok,
        'std_ok': std_ok,
        'both_ok': both_ok
    }


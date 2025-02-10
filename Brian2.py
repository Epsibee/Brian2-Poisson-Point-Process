#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(neurons, rate * b2.Hz, weight=1)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[8]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(neurons, rate * b2.Hz, weight=1)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[10]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(neurons, num_neurons, rate * b2.Hz)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[12]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(neurons, rate * b2.Hz)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[14]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, N=num_neurons, rate=rate*b2.Hz, weight=1*b2.mV)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[16]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1*b2.mV)
    net = b2.Network(neurons, inputs)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    net.add(spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[18]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1*b2.dimless)
    net = b2.Network(neurons, inputs)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    net.add(spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[22]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    net = b2.Network(neurons, inputs)
    b2.run(time * b2.ms)
    spike_monitor = b2.SpikeMonitor(neurons)
    net.add(spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 8
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[1]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time)
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[1]:


import numpy as np
from scipy.stats import poisson # just go with this :)
import brian2 as b2 # does not need to be loaded each time

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        process = np.random.poisson(rate, size=time) # want a poisson process not a sample of poisson random variable
        # brian2 has poisson process generator, "P = PoissonGroup(100, np.arange(100)*Hz + 10*Hz)", figure out parameters
        # P = b2.PoissonGroup(100, np.arange(100)*b2.Hz + 10*b2.Hz), add b2. to Hz and PoissonGroup !!
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            process = np.random.poisson(rate, size=time)
        else:
            process = np.random.poisson(rate, size=time) + correlation * processes[i-1]
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10 # from 10 to 110, search wikipedia regarding rates and 
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[2]:


np.random.poisson(rate, size=time)


# In[3]:


P = b2.PoissonGroup(100, np.arange(100)*b2.Hz + 10*b2.Hz)


# In[5]:


import numpy as np
from scipy.stats import poisson 
import brian2 as b2 

def independent_poisson_processes(num_neurons, rate, time):
    processes = []
    for i in range(num_neurons):
        # Using Brian2's PoissonGroup to generate a Poisson process
        P = b2.PoissonGroup(1, rate*b2.Hz)
        spike_monitor = b2.SpikeMonitor(P)
        b2.run(time * b2.ms)
        spike_times = spike_monitor.spike_trains()[0]
        # Convert spike times to a binary array
        process = np.zeros(time)
        for t in spike_times:
            if t < time:
                process[int(t)] = 1
        processes.append(process)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation):
    processes = []
    for i in range(num_neurons):
        if i == 0:
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = np.zeros(time)
            for t in spike_times:
                if t < time:
                    process[int(t)] = 1
        else:
            # Generate a correlated Poisson process
            prev_process = processes[i-1]
            new_process = np.random.poisson(rate, size=time) + correlation * prev_process
            # Ensure the new process is a binary array
            new_process = np.where(new_process > 0, 1, 0)
            process = new_process
        processes.append(process)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10 
time = 1000
correlation = 0.5

independent_processes = independent_poisson_processes(num_neurons, rate, time)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[i]), np.var(independent_processes[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[i]), np.var(correlated_processes[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))



# In[5]:


import numpy as np
from scipy.stats import poisson 
import brian2 as b2 
import matplotlib.pyplot as plt

def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = np.zeros(time)
            for t in spike_times:
                if t < time:
                    process[int(t)] = 1
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            if i == 0:
                # Using Brian2's PoissonGroup to generate a Poisson process
                P = b2.PoissonGroup(1, rate*b2.Hz)
                spike_monitor = b2.SpikeMonitor(P)
                b2.run(time * b2.ms)
                spike_times = spike_monitor.spike_trains()[0]
                # Convert spike times to a binary array
                process = np.zeros(time)
                for t in spike_times:
                    if t < time:
                        process[int(t)] = 1
            else:
                # Generate a correlated Poisson process
                prev_process = sample_processes[i-1]
                new_process = np.random.poisson(rate, size=time) + correlation * prev_process
                # Ensure the new process is a binary array
                new_process = np.where(new_process > 0, 1, 0)
                process = new_process
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 3
rate = 10 
time = 1000
correlation = 0.5
num_samples = 100

independent_processes = independent_poisson_processes(num_neurons, rate, time, num_samples)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

# Calculate statistics
independent_means = np.mean(independent_processes, axis=0)
independent_vars = np.var(independent_processes, axis=0)

correlated_means = np.mean(correlated_processes, axis=0)
correlated_vars = np.var(correlated_processes, axis=0)

# Plot statistics
plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_means[i], label=f'Neuron {i+1}')
plt.xlabel('Time')
plt.ylabel('Mean')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_means[i], label=f'Neuron {i+1}')
plt.xlabel('Time')
plt.ylabel('Mean')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

# Plot 3D statistics
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_samples):
    ax.plot(independent_processes[i][0], independent_processes[i][1], independent_processes[i][2])
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.set_zlabel('Neuron 3')
plt.title('Independent Poisson Processes')
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_samples):
    ax.plot(correlated_processes[i][0], correlated_processes[i][1], correlated_processes[i][2])
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.set_zlabel('Neuron 3')
plt.title('Correlated Poisson Processes')
plt.show()

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_means[i]), np.mean(independent_vars[i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_means[i]), np.mean(correlated_vars[i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[7]:


import numpy as np
from scipy.stats import poisson 
import brian2 as b2 
import matplotlib.pyplot as plt

def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = np.zeros(int(time))
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, correlation, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            if i == 0:
                # Using Brian2's PoissonGroup to generate a Poisson process
                P = b2.PoissonGroup(1, rate*b2.Hz)
                spike_monitor = b2.SpikeMonitor(P)
                b2.run(time * b2.ms)
                spike_times = spike_monitor.spike_trains()[0]
                # Convert spike times to a binary array
                process = np.zeros(int(time))
                for t in spike_times:
                    if t < time * b2.ms:  # Ensure units match
                        process[int(t / b2.ms)] = 1  # Convert t to unitless
            else:
                # Generate a correlated Poisson process
                prev_process = sample_processes[i-1]
                new_process = np.random.poisson(rate, size=int(time)) + correlation * prev_process
                # Ensure the new process is a binary array
                new_process = np.where(new_process > 0, 1, 0)
                process = new_process
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10 
time = 1000
correlation = 0.5
num_samples = 100

independent_processes = independent_poisson_processes(num_neurons, rate, time, num_samples)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, correlation, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[0][i]), np.var(independent_processes[0][i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[0][i]), np.var(correlated_processes[0][i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))


# In[10]:


from scipy.stats import poisson 
import brian2 as b2 
import matplotlib.pyplot as plt

def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = np.zeros(int(time))
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = np.zeros(int(time))
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = np.sum(independent_procs[:i+1], axis=0)
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10 
time = 1000
num_common = 2
num_samples = 100

independent_processes = independent_poisson_processes(num_neurons, rate, time, num_samples)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[0][i]), np.var(independent_processes[0][i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[0][i]), np.var(correlated_processes[0][i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

# Plot the results
plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()


# In[14]:


from scipy.stats import poisson 
import brian2 as b2 
import matplotlib.pyplot as plt

# **Step 1: Generate Independent Poisson Processes**
def independent_poisson_processes(num_neurons, rate, time, num_samples):

    # **Start with Brian2 to generate the Poisson processes**
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            # Using Brian2's PoissonGroup to generate a Poisson process with b2.
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = np.zeros(int(time))
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match, was having trouble with this (errors)
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            sample_processes.append(process)
            
            # **Summing the independent Poisson processes**
            summing_variable += np.sum(process)
            counts.append(np.sum(process))  # Append count to counts variable
            vector_set_of_counts.append(process)  # Append process to vector set of counts variable
            characterization_of_counts.append(poisson.ppf(np.sum(process), rate * time))  # Append characterization to characterization of counts variable
            
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# **Step 2: Generate Correlated Poisson Processes**
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):

    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = np.zeros(int(time))
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = np.sum(independent_procs[:i+1], axis=0)
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 5
rate = 10 
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

# Print the results
print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(independent_processes[0][i]), np.var(independent_processes[0][i])))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, np.mean(correlated_processes[0][i]), np.var(correlated_processes[0][i])))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))
print("Counts: {}".format(counts))
print("Vector set of counts: {}".format(vector_set_of_counts))
print("Characterization of counts: {}".format(characterization_of_counts))

# Plot the results
plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()


# In[5]:


from scipy.stats import poisson 
import brian2 as b2 
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# **Step 1: Generate Independent Poisson Processes**
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    # **Start with Brian2 to generate the Poisson processes**
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            # Using Brian2's PoissonGroup to generate a Poisson process with b2.
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match, was having trouble with this (errors)
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            sample_processes.append(process)
            
            # **Summing the independent Poisson processes**
            summing_variable += sum(process)
            counts.append(sum(process))  # Append count to counts variable
            vector_set_of_counts.append(process)  # Append process to vector set of counts variable
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))  # Append characterization to characterization of counts variable
            
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# **Step 1a: Calculate Probability Distribution**
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time)
        probabilities.append(probability)
    return probabilities

# **Step 1b: Interpret as a Counting Process**
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# **Step 2: Generate Correlated Poisson Processes**
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# **Step 3: Simulate a Small Network**
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 3
rate = 10 
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("\nProbabilities:", probabilities)
print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(probabilities)
plt.xlabel('Count')
plt.ylabel('Probability')
plt.title('Probability Distribution of Point Counts')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(counts)
plt.xlabel('Sample')
plt.ylabel('Count')
plt.title('Counting Process')
plt.show()

plt.figure(figsize=(8,6))
plt.hist(counts, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Histogram of Counts')
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for i in range(time):
    print(i)
    print([counting_process_2d[j][i] for j in range(num_samples)])
plt.plot([counting_process_2d[j][i] for i in range(time)] for j in range(num_samples))
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for i in range(time):
    print(i)
    print([counting_process_3d[j][i] for j in range(num_samples)])
plt.plot([counting_process_3d[j][i] for i in range(time)] for j in range(num_samples))
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.scatter([counting_process_2d[j][0] for j in range(num_samples)], [counting_process_2d[j][1] for j in range(num_samples)])
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([counting_process_3d[j][0] for j in range(num_samples)], [counting_process_3d[j][1] for j in range(num_samples)], [counting_process_3d[j][2] for j in range(num_samples)])
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
plt.show()


# In[9]:


from scipy.stats import poisson 
import brian2 as b2 
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# **Step 1: Generate Independent Poisson Processes**
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    # **Start with Brian2 to generate the Poisson processes**
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            # Using Brian2's PoissonGroup to generate a Poisson process with b2.
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match, was having trouble with this (errors)
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            sample_processes.append(process)
            
            # **Summing the independent Poisson processes**
            summing_variable += sum(process)
            counts.append(sum(process))  # Append count to counts variable
            vector_set_of_counts.append(process)  # Append process to vector set of counts variable
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))  # Append characterization to characterization of counts variable
            
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# **Step 1a: Calculate Probability Distribution**
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)  # Divide rate by 1000 to avoid numerical underflow
        probabilities.append(probability)
    return probabilities

# **Step 1b: Interpret as a Counting Process**
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# **Step 2: Generate Correlated Poisson Processes**
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# **Step 3: Simulate a Small Network**
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 3
rate = 10 
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("\nProbabilities:", probabilities)
print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(probabilities)
plt.xlabel('Count')
plt.ylabel('Probability')
plt.title('Probability Distribution of Point Counts')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(counts)
plt.xlabel('Sample')
plt.ylabel('Count')
plt.title('Counting Process')
plt.show()

plt.figure(figsize=(8,6))
plt.hist(counts, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Histogram of Counts')
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.scatter([counting_process_2d[j][0] for j in range(num_samples)], [counting_process_2d[j][1] for j in range(num_samples)])
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([counting_process_3d[j][0] for j in range(num_samples)], [counting_process_3d[j][1] for j in range(num_samples)], [counting_process_3d[j][2] for j in range(num_samples)])
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
plt.show()


# In[3]:


from scipy.stats import poisson 
import brian2 as b2 
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# **Step 1: Generate Independent Poisson Processes**
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    # **Start with Brian2 to generate the Poisson processes**
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            # Using Brian2's PoissonGroup to generate a Poisson process with b2.
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match, was having trouble with this (errors)
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            sample_processes.append(process)
            
            # **Summing the independent Poisson processes**
            summing_variable += sum(process)
            counts.append(sum(process))  # Append count to counts variable
            vector_set_of_counts.append(process)  # Append process to vector set of counts variable
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))  # Append characterization to characterization of counts variable
            
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# **Step 1a: Calculate Probability Distribution**
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)  # Divide rate by 1000 to avoid numerical underflow
        probabilities.append(probability)
    return probabilities

# **Step 1b: Interpret as a Counting Process**
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# **Step 2: Generate Correlated Poisson Processes**
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            # Using Brian2's PoissonGroup to generate a Poisson process
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            # Convert spike times to a binary array
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:  # Ensure units match
                    process[int(t / b2.ms)] = 1  # Convert t to unitless
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# **Step 3: Simulate a Small Network**
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

num_neurons = 3
rate = 10 
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("\nProbabilities:", probabilities)
print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(probabilities)
plt.xlabel('Count')
plt.ylabel('Probability')
plt.title('Probability Distribution of Point Counts')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(counts)
plt.xlabel('Sample')
plt.ylabel('Count')
plt.title('Counting Process')
plt.show()

plt.figure(figsize=(8,6))
plt.hist(counts, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Histogram of Counts')
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.scatter([counting_process_2d[j][0] for j in range(num_samples)], [counting_process_2d[j][1] for j in range(num_samples)])
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([counting_process_3d[j][0] for j in range(num_samples)], [counting_process_3d[j][1] for j in range(num_samples)], [counting_process_3d[j][2] for j in range(num_samples)])
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
plt.show()


# In[4]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Improve 2D and 3D Graphs
def generate_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 2: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 3: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 4: Include Brand Networks and Event Neurons
def include_brand_networks_and_event_neurons(num_neurons, rate, time, num_samples):
    processes = generate_poisson_processes(num_neurons, rate, time, num_samples)
    means, covariances = calculate_mean_and_covariance(processes)
    slopes = get_slope(processes)
    return processes, means, covariances, slopes

# Step 5: Vectorize Code and Discretize Time
def vectorize_code_and_discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array(process)
        discretized_processes.append(discretized_process)
    return discretized_processes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_samples = 100

processes, means, covariances, slopes = include_brand_networks_and_event_neurons(num_neurons, rate, time, num_samples)
discretized_processes = vectorize_code_and_discretize_time(processes, time)
plot_neurons_accurately(processes, time)
test_and_refine(processes, time)

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)


# In[5]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Generate Independent Poisson Processes
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# Step 1a: Calculate Probability Distribution
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)
        probabilities.append(probability)
    return probabilities

# Step 1b: Interpret as a Counting Process
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# Step 2: Generate Correlated Poisson Processes
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 3: Simulate a Small Network
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Step 4: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 5: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

means, covariances = calculate_mean_and_covariance(independent_processes)
slopes = get_slope(independent_processes)

plot_neurons_accurately(independent_processes, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("\nProbabilities:", probabilities)
print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(probabilities)
plt.xlabel('Count')
plt.ylabel('Probability')
plt.title('Probability Distribution of Point Counts')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(counts)
plt.xlabel('Sample')
plt.ylabel('Count')
plt.title('Counting Process')
plt.show()

plt.figure(figsize=(8,6))
plt.hist(counts, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Histogram of Counts')
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.scatter([counting_process_2d[j][0] for j in range(num_samples)], [counting_process_2d[j][1] for j in range(num_samples)])
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([counting_process_3d[j][0] for j in range(num_samples)], [counting_process_3d[j][1] for j in range(num_samples)], [counting_process_3d[j][2] for j in range(num_samples)])
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
plt.show()

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)


# In[9]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Generate Independent Poisson Processes
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# Step 1a: Calculate Probability Distribution
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)
        probabilities.append(probability)
    return probabilities

# Step 1b: Interpret as a Counting Process
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# Step 2: Generate Correlated Poisson Processes
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 3: Simulate a Small Network
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Step 4: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 5: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

means, covariances = calculate_mean_and_covariance(independent_processes)
slopes = get_slope(independent_processes)

plot_neurons_accurately(independent_processes, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
slope = np.polyfit(range(time), [counting_process_2d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Two Neurons: ", slope)
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
slope = np.polyfit(range(time), [counting_process_3d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Three Neurons: ", slope)
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]), np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
slope = np.polyfit(range(time), avg_count_2d, 1)[0]
print("Slope of Avg Count of Two Neurons Over Time: ", slope)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
slope = np.polyfit(range(time), avg_count_3d, 1)[0]
print("Slope of Avg Count of Three Neurons Over Time: ", slope)
plt.show()

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)


# In[11]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Generate Independent Poisson Processes
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# Step 1a: Calculate Probability Distribution
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)
        probabilities.append(probability)
    return probabilities

# Step 1b: Interpret as a Counting Process
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# Step 2: Generate Correlated Poisson Processes
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 3: Simulate a Small Network
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Step 4: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 5: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

means, covariances = calculate_mean_and_covariance(independent_processes)
slopes = get_slope(independent_processes)

plot_neurons_accurately(independent_processes, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
slope = np.polyfit(range(time), [counting_process_2d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Two Neurons: ", slope)
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
slope = np.polyfit(range(time), [counting_process_3d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Three Neurons: ", slope)
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]), np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
slope = np.polyfit(range(time), avg_count_2d, 1)[0]
print("Slope of Avg Count of Two Neurons Over Time: ", slope)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
slope = np.polyfit(range(time), avg_count_3d, 1)[0]
print("Slope of Avg Count of Three Neurons Over Time: ", slope)
plt.show()

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)

# Plot a neuron against time (2D) using a single sample standard deviation
plt.figure(figsize=(10,6))
plt.plot(independent_processes[0][0], label='Neuron 1')
plt.fill_between(range(time), independent_processes[0][0] - np.std(independent_processes[0][0]), independent_processes[0][0] + np.std(independent_processes[0][0]), alpha=0.2, label='Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Neuron 1 Over Time with Standard Deviation')
plt.legend()
plt.show()

# Plot two neurons against time (3D) using a covariance matrix that depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(independent_processes[0][0], independent_processes[0][1], np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.set_zlabel('Cumulative Sum of Neuron 3')
plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
plt.show()

# Use the cumsum function to count events or the times function without an event descriptor
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), label='Cumulative Sum of Neuron 1')
plt.xlabel('Time')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of Neuron 1 Over Time')
plt.legend()
plt.show()

# Define a matrix for rates, units, and counts
rates = np.array([[10, 20], [30, 40]])
units = np.array([[1, 2], [3, 4]])
counts = np.array([[100, 200], [300, 400]])

# Vectorize the matrix and choose a subset (dimension) to plot
vectorized_rates = rates.flatten()
vectorized_units = units.flatten()
vectorized_counts = counts.flatten()

plt.figure(figsize=(10,6))
plt.plot(vectorized_rates, vectorized_counts)
plt.xlabel('Rate')
plt.ylabel('Count')
plt.title('Rate vs Count')
plt.show()

# Take the stair function and plot pairs in 3D (time, count1, count2)
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.arange(time), np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Sum of Neuron 1')
ax.set_zlabel('Cumulative Sum of Neuron 2')
plt.title('Cumulative Sums of Two Neurons Over Time')
plt.show()


# In[13]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Generate Independent Poisson Processes
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# Step 1a: Calculate Probability Distribution
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)
        probabilities.append(probability)
    return probabilities

# Step 1b: Interpret as a Counting Process
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# Step 2: Generate Correlated Poisson Processes
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 3: Simulate a Small Network
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Step 4: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 5: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

means, covariances = calculate_mean_and_covariance(independent_processes)
slopes = get_slope(independent_processes)

plot_neurons_accurately(independent_processes, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
slope = np.polyfit(range(time), [counting_process_2d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Two Neurons: ", slope)
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
slope = np.polyfit(range(time), [counting_process_3d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Three Neurons: ", slope)
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]), np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
slope = np.polyfit(range(time), avg_count_2d, 1)[0]
print("Slope of Avg Count of Two Neurons Over Time: ", slope)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
slope = np.polyfit(range(time), avg_count_3d, 1)[0]
print("Slope of Avg Count of Three Neurons Over Time: ", slope)
plt.show()

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)

# Plot a neuron against time (2D) using a single sample standard deviation
plt.figure(figsize=(10,6))
plt.plot(independent_processes[0][0], label='Neuron 1')
plt.fill_between(range(time), independent_processes[0][0] - np.std(independent_processes[0][0]), independent_processes[0][0] + np.std(independent_processes[0][0]), alpha=0.2, label='Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Neuron 1 Over Time with Standard Deviation')
plt.legend()
plt.show()

# Plot two neurons against time (3D) using a covariance matrix that depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(independent_processes[0][0], independent_processes[0][1], np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.set_zlabel('Cumulative Sum of Neuron 3')
plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
plt.show()

# Use the cumsum function to count events or the times function without an event descriptor
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), label='Cumulative Sum of Neuron 1')
plt.xlabel('Time')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of Neuron 1 Over Time')
plt.legend()
plt.show()

# Define a matrix for rates, units, and counts
rates = np.array([[10, 20], [30, 40]])
units = np.array([[1, 2], [3, 4]])
counts = np.array([[100, 200], [300, 400]])

# Vectorize the matrix and choose a subset (dimension) to plot
vectorized_rates = rates.flatten()
vectorized_units = units.flatten()
vectorized_counts = counts.flatten()

plt.figure(figsize=(10,6))
plt.plot(vectorized_rates, vectorized_counts)
plt.xlabel('Rate')
plt.ylabel('Count')
plt.title('Rate vs Count')
plt.show()

# Take the stair function and plot pairs in 3D (time, count1, count2)
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.arange(time), np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Sum of Neuron 1')
ax.set_zlabel('Cumulative Sum of Neuron 2')
plt.title('Cumulative Sums of Two Neurons Over Time')
plt.show()

# Step 2: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Calculate the mean of the neurons using one sample standard deviation
mean_with_std = np.mean(independent_processes[0][0]) + np.std(independent_processes[0][0])
print("Mean with standard deviation: ", mean_with_std)

# Calculate the covariance matrix for the neurons, which depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
print("Covariance matrix: ", cov_matrix)

# Step 4: Include Brand Networks and Event Neurons
def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Incorporate brand networks and event neurons into the program
brand_spike_times = brand_networks(num_neurons, rate, time)
print("Brand spike times: ", brand_spike_times)

# Use a Poisson group with n neurons, allowing for one rate or multiple rates
poisson_group = b2.PoissonGroup(num_neurons, rate*b2.Hz)
spike_monitor = b2.SpikeMonitor(poisson_group)
b2.run(time * b2.ms)
poisson_spike_times = spike_monitor.spike_trains()
print("Poisson spike times: ", poisson_spike_times)

# Step 5: Vectorize Code and Discretize Time
def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

# Replace for loops with vectorized code
vectorized_processes = vectorize_code(independent_processes)
print("Vectorized processes: ", vectorized_processes)

# Check if the return value is also vectorized and convert it to a single value if necessary
def check_vectorization(processes):
    if isinstance(processes, list):
        return np.array(processes)
    else:
        return processes

# Discretize the time and events to plot the neurons accurately
def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

# Sample the spikes to some resolution and tag each event with a Poisson process
def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Sample the spikes
sampled_processes = sample_spikes(independent_processes, time)
print("Sampled processes: ", sampled_processes)


# In[1]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

NUM_NEURONS = 3
RATE = 10
TIME = 1000
NUM_COMMON = 2
NUM_SAMPLES = 100

class NeuralNetworkSimulator:
    def __init__(self, num_neurons, rate, time, num_common, num_samples):
        self.num_neurons = num_neurons
        self.rate = rate
        self.time = time
        self.num_common = num_common
        self.num_samples = num_samples

    def generate_independent_poisson_processes(self):
        try:
            processes = []
            summing_variable = 0
            counts = []
            vector_set_of_counts = []
            characterization_of_counts = []

            for _ in range(self.num_samples):
                sample_processes = []
                for i in range(self.num_neurons):
                    P = b2.PoissonGroup(1, self.rate * b2.Hz)
                    spike_monitor = b2.SpikeMonitor(P)
                    b2.run(self.time * b2.ms)
                    spike_times = spike_monitor.spike_trains()[0]
                    process = [0] * int(self.time)
                    for t in spike_times:
                        if t < self.time * b2.ms:
                            process[int(t / b2.ms)] = 1
                    sample_processes.append(process)

                    summing_variable += sum(process)
                    counts.append(sum(process))
                    vector_set_of_counts.append(process)
                    characterization_of_counts.append(poisson.ppf(sum(process), self.rate * self.time))

                processes.append(sample_processes)

            return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts
        except Exception as e:
            print(f"Error generating independent Poisson processes: {e}")
            return None

    def generate_correlated_poisson_processes(self):
        try:
            processes = []
            for _ in range(self.num_samples):
                sample_processes = []
                independent_procs = []
                for i in range(self.num_neurons + self.num_common):
                    P = b2.PoissonGroup(1, self.rate * b2.Hz)
                    spike_monitor = b2.SpikeMonitor(P)
                    b2.run(self.time * b2.ms)
                    spike_times = spike_monitor.spike_trains()[0]
                    process = [0] * int(self.time)
                    for t in spike_times:
                        if t < self.time * b2.ms:
                            process[int(t / b2.ms)] = 1
                    independent_procs.append(process)

                for i in range(self.num_neurons):
                    if i < self.num_common:
                        process = [sum(x) for x in zip(*independent_procs[:i + 1])]
                    else:
                        process = independent_procs[i + self.num_common]
                    sample_processes.append(process)
                processes.append(sample_processes)

            return processes
        except Exception as e:
            print(f"Error generating correlated Poisson processes: {e}")
            return None

    def simulate_small_network(self):
        try:
            neurons = b2.NeuronGroup(self.num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
            inputs = b2.PoissonInput(target=neurons, target_var='v', N=self.num_neurons, rate=self.rate * b2.Hz, weight=1)
            spike_monitor = b2.SpikeMonitor(neurons)
            net = b2.Network(neurons, inputs, spike_monitor)
            b2.run(self.time * b2.ms)
            return spike_monitor.spike_trains()
        except Exception as e:
            print(f"Error simulating small network: {e}")
            return None

    def calculate_mean_and_covariance(self, processes):
        try:
            means = []
            covariances = []
            for i in range(len(processes[0])):
                mean = np.mean([process[i] for process in processes])
                covariance = np.cov([process[i] for process in processes])
                means.append(mean)
                covariances.append(covariance)

            return means, covariances
        except Exception as e:
            print(f"Error calculating mean and covariance: {e}")
            return None

    def plot_neurons_accurately(self, processes, time):
        try:
            fig = plt.figure(figsize=(10, 6))
            for i in range(len(processes[0])):
                plt.plot(processes[0][i], label=f'Neuron {i}')
            plt.xlabel('Time')
            plt.ylabel('Spike')
            plt.title('Neurons Over Time')
            plt.legend()
            plt.show()

            fig = plt.figure(figsize=(10, 6))
            plt.scatter(processes[0][0], processes[0][1])
            plt.xlabel('Neuron 1')
            plt.ylabel('Neuron 2')
            plt.title('Neuron 1 vs Neuron 2')
            plt.show()

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(processes[0][0], processes[0][1], processes[0][2])
            ax.set_xlabel('Neuron 1')
            ax.set_ylabel('Neuron 2')
            ax.set_zlabel('Neuron 3')
            plt.title('Neurons in 3D')
            plt.show()
        except Exception as e:
            print(f"Error plotting neurons accurately: {e}")

def main():
    simulator = NeuralNetworkSimulator(NUM_NEURONS, RATE, TIME, NUM_COMMON, NUM_SAMPLES)

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = simulator.generate_independent_poisson_processes()
    correlated_processes = simulator.generate_correlated_poisson_processes()
    network_spike_times = simulator.simulate_small_network()

    means, covariances = simulator.calculate_mean_and_covariance(independent_processes)
    simulator.plot_neurons_accurately(independent_processes, TIME)

    print("Independent Poisson processes:")
    for i in range(NUM_NEURONS):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(NUM_NEURONS):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(NUM_NEURONS):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

if __name__ == "__main__":
    main()


# In[ ]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

NUM_NEURONS = 3
RATE = 10
TIME = 1000
NUM_COMMON = 2
NUM_SAMPLES = 100

def _generate_poisson_group(rate):
    return b2.PoissonGroup(1, rate * b2.Hz)

def _run_simulation(time):
    b2.run(time * b2.ms)

def _get_spike_times(spike_monitor):
    return spike_monitor.spike_trains()[0]

def _create_process(spike_times, time):
    process = [0] * int(time)
    for t in spike_times:
        if t < time * b2.ms:
            process[int(t / b2.ms)] = 1
    return process

def generate_independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []

    for _ in range(num_samples):
        sample_processes = []
        net = b2.Network()
        poisson_groups = []  # Store PoissonGroup objects
        spike_monitors = []  # Store SpikeMonitor objects
        
        for i in range(num_neurons):
            P = _generate_poisson_group(rate)
            net.add(P)
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)
            poisson_groups.append(P)  # Store the PoissonGroup
            spike_monitors.append(spike_monitor)  # Store the SpikeMonitor
            
        _run_simulation(time)
        
        for spike_monitor in spike_monitors:
            spike_times = _get_spike_times(spike_monitor)
            process = _create_process(spike_times, time)
            sample_processes.append(process)

            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))

        processes.append(sample_processes)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts


def generate_correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = _generate_poisson_group(rate)
            spike_monitor = b2.SpikeMonitor(P)
            _run_simulation(time)
            spike_times = _get_spike_times(spike_monitor)
            process = _create_process(spike_times, time)
            independent_procs.append(process)

        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i + 1])]
            else:
                process = independent_procs[i + num_common]
            sample_processes.append(process)
        processes.append(sample_processes)

    return processes

def simulate_small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate * b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    _run_simulation(time)
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)

    return means, covariances

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10, 6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = generate_independent_poisson_processes(NUM_NEURONS, RATE, TIME, NUM_SAMPLES)
correlated_processes = generate_correlated_poisson_processes(NUM_NEURONS, RATE, TIME, NUM_COMMON, NUM_SAMPLES)
network_spike_times = simulate_small_network(NUM_NEURONS, RATE, TIME)

means, covariances = calculate_mean_and_covariance(independent_processes)
plot_neurons_accurately(independent_processes, TIME)

print("Independent Poisson processes:")
for i in range(NUM_NEURONS):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(NUM_NEURONS):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(NUM_NEURONS):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))


# In[16]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Generate Independent Poisson Processes
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# Step 1a: Calculate Probability Distribution
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)
        probabilities.append(probability)
    return probabilities

# Step 1b: Interpret as a Counting Process
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# Step 2: Generate Correlated Poisson Processes
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 3: Simulate a Small Network
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Step 4: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 5: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

means, covariances = calculate_mean_and_covariance(independent_processes)
slopes = get_slope(independent_processes)

plot_neurons_accurately(independent_processes, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
slope = np.polyfit(range(time), [counting_process_2d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Two Neurons: ", slope)
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
slope = np.polyfit(range(time), [counting_process_3d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Three Neurons: ", slope)
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]), np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
slope = np.polyfit(range(time), avg_count_2d, 1)[0]
print("Slope of Avg Count of Two Neurons Over Time: ", slope)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
slope = np.polyfit(range(time), avg_count_3d, 1)[0]
print("Slope of Avg Count of Three Neurons Over Time: ", slope)
plt.show()

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)

# Plot a neuron against time (2D) using a single sample standard deviation
plt.figure(figsize=(10,6))
plt.plot(independent_processes[0][0], label='Neuron 1')
plt.fill_between(range(time), independent_processes[0][0] - np.std(independent_processes[0][0]), independent_processes[0][0] + np.std(independent_processes[0][0]), alpha=0.2, label='Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Neuron 1 Over Time with Standard Deviation')
plt.legend()
plt.show()

# Plot two neurons against time (3D) using a covariance matrix that depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(independent_processes[0][0], independent_processes[0][1], np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.set_zlabel('Cumulative Sum of Neuron 3')
plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
plt.show()

# Use the cumsum function to count events or the times function without an event descriptor
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), label='Cumulative Sum of Neuron 1')
plt.xlabel('Time')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of Neuron 1 Over Time')
plt.legend()
plt.show()

# Define a matrix for rates, units, and counts
rates = np.array([[10, 20], [30, 40]])
units = np.array([[1, 2], [3, 4]])
counts = np.array([[100, 200], [300, 400]])

# Vectorize the matrix and choose a subset (dimension) to plot
vectorized_rates = rates.flatten()
vectorized_units = units.flatten()
vectorized_counts = counts.flatten()

plt.figure(figsize=(10,6))
plt.plot(vectorized_rates, vectorized_counts)
plt.xlabel('Rate')
plt.ylabel('Count')
plt.title('Rate vs Count')
plt.show()

# Take the stair function and plot pairs in 3D (time, count1, count2)
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.arange(time), np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Sum of Neuron 1')
ax.set_zlabel('Cumulative Sum of Neuron 2')
plt.title('Cumulative Sums of Two Neurons Over Time')
plt.show()

# Step 2: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Calculate the mean of the neurons using one sample standard deviation
mean_with_std = np.mean(independent_processes[0][0]) + np.std(independent_processes[0][0])
print("Mean with standard deviation: ", mean_with_std)

# Calculate the covariance matrix for the neurons, which depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
print("Covariance matrix: ", cov_matrix)

# Step 4: Include Brand Networks and Event Neurons
def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Incorporate brand networks and event neurons into the program
brand_spike_times = brand_networks(num_neurons, rate, time)
print("Brand spike times: ", brand_spike_times)

# Use a Poisson group with n neurons, allowing for one rate or multiple rates
poisson_group = b2.PoissonGroup(num_neurons, rate*b2.Hz)
spike_monitor = b2.SpikeMonitor(poisson_group)
b2.run(time * b2.ms)
poisson_spike_times = spike_monitor.spike_trains()
print("Poisson spike times: ", poisson_spike_times)

# Step 5: Vectorize Code and Discretize Time
def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

# Replace for loops with vectorized code
vectorized_processes = vectorize_code(independent_processes)

# Check if the return value is also vectorized and convert it to a single value if necessary
def check_vectorization(processes):
    if isinstance(processes, list):
        return np.array(processes)
    else:
        return processes

# Discretize the time and events to plot the neurons accurately
def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

# Sample the spikes to some resolution and tag each event with a Poisson process
def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Sample the spikes
sampled_processes = sample_spikes(independent_processes, time)
print("Sampled processes: ", sampled_processes)


# In[18]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = 'numpy'
import matplotlib.pyplot as plt

# Generate Independent Poisson Processes

def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = np.zeros((num_samples, num_neurons, time), dtype=int)
    counts = np.zeros((num_samples, num_neurons), dtype=int)

    for sample in range(num_samples):
        P = b2.PoissonGroup(num_neurons, rate * b2.Hz)
        spike_monitor = b2.SpikeMonitor(P)
        b2.run(time * b2.ms)

        # Update processes and counts
        spike_times = spike_monitor.spike_trains()
        for i in range(num_neurons):
            for t in spike_times[i]:
                if t < time * b2.ms:
                    processes[sample, i, int(t / b2.ms)] = 1
            counts[sample, i] = spike_monitor.count[i]

    return processes, counts

# Calculate Probability Distribution

def calculate_probability_distribution(counts, rate, time):
    return poisson.pmf(counts, rate * time / 1000)

# Generate Correlated Poisson Processes

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    independent_processes, _ = independent_poisson_processes(num_neurons + num_common, rate, time, num_samples)
    correlated_processes = np.zeros((num_samples, num_neurons, time), dtype=int)

    for sample in range(num_samples):
        for i in range(num_neurons):
            if i < num_common:
                correlated_processes[sample, i] = np.sum(independent_processes[sample, :i + 1], axis=0)
            else:
                correlated_processes[sample, i] = independent_processes[sample, i + num_common]

    return correlated_processes

# Simulate a Small Network

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate * b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Calculate Statistics: Mean and Covariance

def calculate_mean_and_covariance(processes):
    means = np.mean(processes, axis=(0, 1))
    covariances = np.cov(processes.reshape(-1, processes.shape[1]), rowvar=False)
    return means, covariances

# Plot Neurons

def plot_processes(processes, title):
    plt.figure(figsize=(10, 6))
    for i in range(processes.shape[1]):
        plt.plot(processes[0][i], label=f'Neuron {i + 1}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title(title)
    plt.legend()
    plt.show()

# Main Execution
num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

# Independent Processes
independent_processes, counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)

# Correlated Processes
correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)

# Small Network
network_spike_times = small_network(num_neurons, rate, time)

# Statistics
means, covariances = calculate_mean_and_covariance(independent_processes)

# Plotting
plot_processes(independent_processes, 'Independent Poisson Processes')
plot_processes(correlated_processes, 'Correlated Poisson Processes')

# Outputs
print("Means:", means)
print("Covariances:", covariances)


# In[22]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Generate Independent Poisson Processes
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

# Step 1a: Calculate Probability Distribution
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)
        probabilities.append(probability)
    return probabilities

# Step 1b: Interpret as a Counting Process
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# Step 2: Generate Correlated Poisson Processes
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 3: Simulate a Small Network
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Step 4: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 5: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

means, covariances = calculate_mean_and_covariance(independent_processes)
slopes = get_slope(independent_processes)

plot_neurons_accurately(independent_processes, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
slope = np.polyfit(range(time), [counting_process_2d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Two Neurons: ", slope)
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
slope = np.polyfit(range(time), [counting_process_3d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Three Neurons: ", slope)
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]), np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
slope = np.polyfit(range(time), avg_count_2d, 1)[0]
print("Slope of Avg Count of Two Neurons Over Time: ", slope)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
slope = np.polyfit(range(time), avg_count_3d, 1)[0]
print("Slope of Avg Count of Three Neurons Over Time: ", slope)
plt.show()

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)

# Plot a neuron against time (2D) using a single sample standard deviation
plt.figure(figsize=(10,6))
plt.plot(independent_processes[0][0], label='Neuron 1')
plt.fill_between(range(time), independent_processes[0][0] - np.std(independent_processes[0][0]), independent_processes[0][0] + np.std(independent_processes[0][0]), alpha=0.2, label='Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Neuron 1 Over Time with Standard Deviation')
plt.legend()
plt.show()

# Plot two neurons against time (3D) using a covariance matrix that depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(independent_processes[0][0], independent_processes[0][1], np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.set_zlabel('Cumulative Sum of Neuron 3')
plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
plt.show()

# Use the cumsum function to count events or the times function without an event descriptor
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), label='Cumulative Sum of Neuron 1')
plt.xlabel('Time')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of Neuron 1 Over Time')
plt.legend()
plt.show()

# Define a matrix for rates, units, and counts
rates = np.array([[10, 20], [30, 40]])
units = np.array([[1, 2], [3, 4]])
counts = np.array([[100, 200], [300, 400]])

# Vectorize the matrix and choose a subset (dimension) to plot
vectorized_rates = rates.flatten()
vectorized_units = units.flatten()
vectorized_counts = counts.flatten()

plt.figure(figsize=(10,6))
plt.plot(vectorized_rates, vectorized_counts)
plt.xlabel('Rate')
plt.ylabel('Count')
plt.title('Rate vs Count')
plt.show()

# Take the stair function and plot pairs in 3D (time, count1, count2)
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.arange(time), np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Sum of Neuron 1')
ax.set_zlabel('Cumulative Sum of Neuron 2')
plt.title('Cumulative Sums of Two Neurons Over Time')
plt.show()

# Step 2: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Calculate the mean of the neurons using one sample standard deviation
mean_with_std = np.mean(independent_processes[0][0]) + np.std(independent_processes[0][0])
print("Mean with standard deviation: ", mean_with_std)

# Calculate the covariance matrix for the neurons, which depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
print("Covariance matrix: ", cov_matrix)

# Step 4: Include Brand Networks and Event Neurons
def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Incorporate brand networks and event neurons into the program
brand_spike_times = brand_networks(num_neurons, rate, time)
print("Brand spike times: ", brand_spike_times)

# Use a Poisson group with n neurons, allowing for one rate or multiple rates
poisson_group = b2.PoissonGroup(num_neurons, rate*b2.Hz)
spike_monitor = b2.SpikeMonitor(poisson_group)
b2.run(time * b2.ms)
poisson_spike_times = spike_monitor.spike_trains()
print("Poisson spike times: ", poisson_spike_times)

# Step 5: Vectorize Code and Discretize Time
def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

# Replace for loops with vectorized code
vectorized_processes = vectorize_code(independent_processes)

# Check if the return value is also vectorized and convert it to a single value if necessary
def check_vectorization(processes):
    if isinstance(processes, list):
        return np.array(processes)
    else:
        return processes

# Discretize the time and events to plot the neurons accurately
def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

# Sample the spikes to some resolution and tag each event with a Poisson process
def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Sample the spikes
sampled_processes = sample_spikes(independent_processes, time)
print("Sampled processes: ", sampled_processes)


# In[26]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Step 1: Generate Independent Poisson Processes
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        net = b2.Network()  # Create a new network for each sample
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            net.add(P)  # Add the PoissonGroup to the network
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)  # Add the SpikeMonitor to the network
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
        net.restore()  # Restore the network after each sample
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts


# Step 1a: Calculate Probability Distribution
def calculate_probability_distribution(counts, rate, time):
    probabilities = []
    for count in counts:
        probability = poisson.pmf(count, rate * time / 1000)
        probabilities.append(probability)
    return probabilities

# Step 1b: Interpret as a Counting Process
def interpret_as_counting_process(counts, rate, time):
    expected_value = sum(counts) / len(counts)
    return expected_value

# Step 2: Generate Correlated Poisson Processes
def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

# Step 3: Simulate a Small Network
def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Step 4: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Step 5: Get Slope for Line on Last Graphs
def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

# Step 6: Plot Neurons Accurately
def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

# Step 7: Test and Refine
def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

num_neurons = 3
rate = 10
time = 1000
num_common = 2
num_samples = 100

independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
probabilities = calculate_probability_distribution(counts, rate, time)
expected_value = interpret_as_counting_process(counts, rate, time)

correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
network_spike_times = small_network(num_neurons, rate, time)

means, covariances = calculate_mean_and_covariance(independent_processes)
slopes = get_slope(independent_processes)

plot_neurons_accurately(independent_processes, time)

print("Independent Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

print("\nCorrelated Poisson processes:")
for i in range(num_neurons):
    print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

print("\nNetwork spike times:")
for i in range(num_neurons):
    print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

print("\nSumming variable: {}".format(summing_variable))

print("Expected value of counting process:", expected_value)

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(independent_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Independent Poisson Processes')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(num_neurons):
    plt.plot(correlated_processes[0][i], label=f'Neuron {i}')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Correlated Poisson Processes')
plt.legend()
plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
slope = np.polyfit(range(time), [counting_process_2d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Two Neurons: ", slope)
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
slope = np.polyfit(range(time), [counting_process_3d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Three Neurons: ", slope)
plt.show()

# 2D plot of counting process for two neurons
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
plt.xlabel('Count of Neuron 1')
plt.ylabel('Count of Neuron 2')
plt.title('2D Plot of Counting Process for Two Neurons')
plt.show()

# 3D plot of counting process for three neurons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]), np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Count of Neuron 1')
ax.set_ylabel('Count of Neuron 2')
ax.set_zlabel('Count of Neuron 3')
plt.title('3D Plot of Counting Process for Three Neurons')
plt.show()

# Calculate stats as a function of time
avg_count_2d = [sum([counting_process_2d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]
avg_count_3d = [sum([counting_process_3d[j][i] for j in range(num_samples)]) / num_samples for i in range(time)]

# Plot avg count as a function of time (should be a straight line with slope equal to the rate
plt.figure(figsize=(10,6))
plt.plot(avg_count_2d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Two Neurons Over Time')
slope = np.polyfit(range(time), avg_count_2d, 1)[0]
print("Slope of Avg Count of Two Neurons Over Time: ", slope)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(avg_count_3d)
plt.xlabel('Time')
plt.ylabel('Avg Count')
plt.title('Avg Count of Three Neurons Over Time')
slope = np.polyfit(range(time), avg_count_3d, 1)[0]
print("Slope of Avg Count of Three Neurons Over Time: ", slope)
plt.show()

print("Means:", means)
print("Covariances:", covariances)
print("Slopes:", slopes)

# Plot a neuron against time (2D) using a single sample standard deviation
plt.figure(figsize=(10,6))
plt.plot(independent_processes[0][0], label='Neuron 1')
plt.fill_between(range(time), independent_processes[0][0] - np.std(independent_processes[0][0]), independent_processes[0][0] + np.std(independent_processes[0][0]), alpha=0.2, label='Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.title('Neuron 1 Over Time with Standard Deviation')
plt.legend()
plt.show()

# Plot two neurons against time (3D) using a covariance matrix that depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(independent_processes[0][0], independent_processes[0][1], np.cumsum(independent_processes[0][2]))
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.set_zlabel('Cumulative Sum of Neuron 3')
plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
plt.show()

# Use the cumsum function to count events or the times function without an event descriptor
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(independent_processes[0][0]), label='Cumulative Sum of Neuron 1')
plt.xlabel('Time')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of Neuron 1 Over Time')
plt.legend()
plt.show()

# Define a matrix for rates, units, and counts
rates = np.array([[10, 20], [30, 40]])
units = np.array([[1, 2], [3, 4]])
counts = np.array([[100, 200], [300, 400]])

# Vectorize the matrix and choose a subset (dimension) to plot
vectorized_rates = rates.flatten()
vectorized_units = units.flatten()
vectorized_counts = counts.flatten()

plt.figure(figsize=(10,6))
plt.plot(vectorized_rates, vectorized_counts)
plt.xlabel('Rate')
plt.ylabel('Count')
plt.title('Rate vs Count')
plt.show()

# Take the stair function and plot pairs in 3D (time, count1, count2)
plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.arange(time), np.cumsum(independent_processes[0][0]), np.cumsum(independent_processes[0][1]))
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Sum of Neuron 1')
ax.set_zlabel('Cumulative Sum of Neuron 2')
plt.title('Cumulative Sums of Two Neurons Over Time')
plt.show()

# Step 2: Calculate Mean and Covariance Matrix
def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Calculate the mean of the neurons using one sample standard deviation
mean_with_std = np.mean(independent_processes[0][0]) + np.std(independent_processes[0][0])
print("Mean with standard deviation: ", mean_with_std)

# Calculate the covariance matrix for the neurons, which depends on dimensionality
cov_matrix = np.cov([independent_processes[0][0], independent_processes[0][1]])
print("Covariance matrix: ", cov_matrix)

# Step 4: Include Brand Networks and Event Neurons
def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

# Incorporate brand networks and event neurons into the program
brand_spike_times = brand_networks(num_neurons, rate, time)
print("Brand spike times: ", brand_spike_times)

# Use a Poisson group with n neurons, allowing for one rate or multiple rates
poisson_group = b2.PoissonGroup(num_neurons, rate*b2.Hz)
spike_monitor = b2.SpikeMonitor(poisson_group)
b2.run(time * b2.ms)
poisson_spike_times = spike_monitor.spike_trains()
print("Poisson spike times: ", poisson_spike_times)

# Step 5: Vectorize Code and Discretize Time
def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

# Replace for loops with vectorized code
vectorized_processes = vectorize_code(independent_processes)

# Check if the return value is also vectorized and convert it to a single value if necessary
def check_vectorization(processes):
    if isinstance(processes, list):
        return np.array(processes)
    else:
        return processes

# Discretize the time and events to plot the neurons accurately
def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

# Sample the spikes to some resolution and tag each event with a Poisson process
def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Sample the spikes
sampled_processes = sample_spikes(independent_processes, time)
print("Sampled processes: ", sampled_processes)


# In[28]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons, rate, time, num_samples):
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []
    
    for _ in range(num_samples):
        sample_processes = []
        net = b2.Network()  
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            net.add(P)  
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)  
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)
            
            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.ppf(sum(process), rate * time))
        
        processes.append(sample_processes)
        net.restore()  
    
    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            spike_monitor = b2.SpikeMonitor(P)
            b2.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    b2.run(time * b2.ms)
    return spike_monitor.spike_trains()

def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    brand_spike_times = brand_networks(num_neurons, rate, time)
    print("Brand spike times: ", brand_spike_times)

    vectorized_processes = vectorize_code(independent_processes)
    discretized_processes = discretize_time(independent_processes, time)
    sampled_processes = sample_spikes(independent_processes, time)

    test_and_refine(independent_processes, time)


# In[32]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []

    for _ in range(num_samples):
        sample_processes = []
        net = b2.Network()
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate * b2.Hz)
            net.add(P)
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)
            net.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            sample_processes.append(process)

            summing_variable += sum(process)
            counts.append(sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.pmf(sum(process), rate * time))

        processes.append(sample_processes)
        net.restore()

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        net = b2.Network()  # Create a new network for each sample
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            net.add(P)  # Add the PoissonGroup to the network
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)  # Add the SpikeMonitor to the network
            net.run(time * b2.ms)  # Run the network instead of b2.run
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
        net.restore()  # Restore the network after each sample
    
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    brand_spike_times = brand_networks(num_neurons, rate, time)
    print("Brand spike times: ", brand_spike_times)

    vectorized_processes = vectorize_code(independent_processes)
    discretized_processes = discretize_time(independent_processes, time)
    sampled_processes = sample_spikes(independent_processes, time)

    test_and_refine(independent_processes, time)


# In[36]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []

    for _ in range(num_samples):
        sample_processes = []
        net = b2.Network()
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate * b2.Hz)
            net.add(P)
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)
            net.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = np.zeros(int(time))
            process[np.array([int(t / b2.ms) for t in spike_times if t < time * b2.ms])] = 1
            sample_processes.append(process)

            summing_variable += np.sum(process)
            counts.append(np.sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.pmf(np.sum(process), rate * time))

        processes.append(sample_processes)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        net = b2.Network()  # Create a new network for each sample
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            net.add(P)  # Add the PoissonGroup to the network
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)  # Add the SpikeMonitor to the network
            net.run(time * b2.ms)  # Run the network instead of b2.run
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    brand_spike_times = brand_networks(num_neurons, rate, time)
    print("Brand spike times: ", brand_spike_times)

    vectorized_processes = vectorize_code(independent_processes)
    discretized_processes = discretize_time(independent_processes, time)
    sampled_processes = sample_spikes(independent_processes, time)

    test_and_refine(independent_processes, time)


# In[42]:


import numpy as np
from scipy.stats import poisson
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    processes = []
    summing_variable = 0
    counts = []
    vector_set_of_counts = []
    characterization_of_counts = []

    for _ in range(num_samples):
        sample_processes = []
        net = b2.Network()
        for i in range(num_neurons):
            P = b2.PoissonGroup(1, rate * b2.Hz)
            net.add(P)
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)
            net.run(time * b2.ms)
            spike_times = spike_monitor.spike_trains()[0]
            process = np.zeros(int(time))
            if len(spike_times) > 0:
                indices = np.array([int(t / b2.ms) for t in spike_times if t < time * b2.ms], dtype=int)
                if indices.size > 0:  
                    process[indices] = 1

            sample_processes.append(process)

            summing_variable += np.sum(process)
            counts.append(np.sum(process))
            vector_set_of_counts.append(process)
            characterization_of_counts.append(poisson.pmf(np.sum(process), rate * time))

        processes.append(sample_processes)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    processes = []
    for _ in range(num_samples):
        sample_processes = []
        independent_procs = []
        net = b2.Network()  # Create a new network for each sample
        for i in range(num_neurons + num_common):
            P = b2.PoissonGroup(1, rate*b2.Hz)
            net.add(P)  # Add the PoissonGroup to the network
            spike_monitor = b2.SpikeMonitor(P)
            net.add(spike_monitor)  # Add the SpikeMonitor to the network
            net.run(time * b2.ms)  # Run the network instead of b2.run
            spike_times = spike_monitor.spike_trains()[0]
            process = [0] * int(time)
            for t in spike_times:
                if t < time * b2.ms:
                    process[int(t / b2.ms)] = 1
            independent_procs.append(process)
        
        for i in range(num_neurons):
            if i < num_common:
                process = [sum(x) for x in zip(*independent_procs[:i+1])]
            else:
                process = independent_procs[i+num_common]
            sample_processes.append(process)
        processes.append(sample_processes)
    
    return processes

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,6))
    plt.scatter(processes[0][0], processes[0][1])
    plt.xlabel('Neuron 1')
    plt.ylabel('Neuron 2')
    plt.title('Neuron 1 vs Neuron 2')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(processes[0][0], processes[0][1], processes[0][2])
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Neuron 3')
    plt.title('Neurons in 3D')
    plt.show()

def test_and_refine(processes, time):
    # Test discrete time and continuous time for Poisson process
    print("Testing discrete time and continuous time for Poisson process...")
    
    # Use binomial cross-section for Bernoulli process
    print("Using binomial cross-section for Bernoulli process...")
    
    # Make graph color equal to length of time
    print("Making graph color equal to length of time...")
    
    # Refine code to achieve slowly increasing average
    print("Refining code to achieve slowly increasing average...")

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    brand_spike_times = brand_networks(num_neurons, rate, time)
    print("Brand spike times: ", brand_spike_times)

    vectorized_processes = vectorize_code(independent_processes)
    discretized_processes = discretize_time(independent_processes, time)
    sampled_processes = sample_spikes(independent_processes, time)

    test_and_refine(independent_processes, time)


# In[50]:


import numpy as np
from scipy.stats import poisson, binom
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate * b2.Hz) for _ in range(num_neurons * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    processes = np.zeros((num_samples, num_neurons, int(time)))
    spike_indices = np.array([np.array([int(t / b2.ms) for t in spike_train if t < time * b2.ms], dtype=int) for spike_train in spike_trains])
    max_spike_count = max(len(indices) for indices in spike_indices)
    padded_spike_indices = np.fromiter((np.pad(indices, (0, max_spike_count - len(indices)), mode='constant', constant_values=-1) for indices in spike_indices), dtype=object)
    sample_indices, neuron_indices = np.divmod(np.arange(num_neurons * num_samples), num_neurons)
    mask = np.fromiter((indices != -1 for indices in padded_spike_indices), dtype=object)
    for i, indices in enumerate(padded_spike_indices):
        processes[sample_indices[i], neuron_indices[i], indices] = mask[i]

    # Calculate summing variable, counts, vector set of counts, and characterization of counts
    summing_variable = np.sum(processes)
    counts = np.sum(processes, axis=(0, 2)).flatten()
    vector_set_of_counts = processes.reshape(-1, int(time))
    characterization_of_counts = poisson.pmf(np.sum(processes, axis=(0, 2)), rate * time)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts


def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate*b2.Hz) for _ in range((num_neurons + num_common) * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    independent_procs = np.zeros((num_samples, num_neurons + num_common, int(time)))
    spike_indices = np.array([np.array([int(t / b2.ms) for t in spike_train if t < time * b2.ms], dtype=int) for spike_train in spike_trains])
    sample_indices, neuron_indices = np.divmod(np.arange((num_neurons + num_common) * num_samples), num_neurons + num_common)
    mask = np.zeros((num_samples * (num_neurons + num_common), int(time)), dtype='bool')
    mask[np.arange(len(spike_indices)), spike_indices] = True
    independent_procs[sample_indices[:, None], neuron_indices[:, None], np.arange(int(time))] = mask

    # Generate correlated processes
    correlated_procs = np.zeros((num_samples, num_neurons, int(time)))
    correlated_procs[:, :num_common, :] = np.cumsum(independent_procs[:, :num_common, :], axis=1)
    correlated_procs[:, num_common:, :] = independent_procs[:, num_common:, :]

    return correlated_procs

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    # 2D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        plt.plot(range(start, end), processes[0][0][start:end], color=colors[i], label=f'Time Section {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('2D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # 3D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        ax.plot(processes[0][0][start:end], processes[0][1][start:end], np.arange(start, end), color=colors[i], label=f'Time Section {i+1}')
    ax.set_xlabel('Neuron 1 Count')
    ax.set_ylabel('Neuron 2 Count')
    ax.set_zlabel('Time')
    plt.title('3D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # Discrete time
    discrete_time_processes = np.sum(processes, axis=2)
    
    # Continuous time
    continuous_time_processes = np.cumsum(processes, axis=2)
    
    # Plot discrete and continuous time processes
    plt.figure(figsize=(10, 6))
    plt.plot(discrete_time_processes[0, 0], label='Discrete Time')
    plt.plot(continuous_time_processes[0, 0], label='Continuous Time')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Discrete and Continuous Time Poisson Processes')
    plt.legend()
    plt.show()
    
    # Use binomial cross-section for Bernoulli process
    bernoulli_process = np.random.binomial(1, 0.5, size=(len(processes), len(processes[0]), len(processes[0][0])))
    binomial_cross_section = np.sum(bernoulli_process, axis=2)
    plt.figure(figsize=(10, 6))
    plt.plot(binomial_cross_section[0, 0], label='Binomial Cross-Section')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Binomial Cross-Section of Bernoulli Process')
    plt.legend()
    plt.show()

    # Refine code to achieve slowly increasing average

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, rate, time, num_rates=None):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')

    if num_rates is None:
        # Use a single rate for all neurons
        inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    else:
        # Use multiple rates for different neurons
        rates = np.random.uniform(0, rate, size=num_rates)
        inputs = []
        for i in range(num_neurons):
            input_rate = rates[i % num_rates] * b2.Hz
            input_ = b2.PoissonInput(target=neurons, target_var='v', N=1, rate=input_rate, weight=1)
            inputs.append(input_)

    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

def is_vectorized(processes):
    return all(isinstance(process, np.ndarray) for process in processes)

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

def tag_events_with_poisson(processes, rate):
    tagged_processes = []
    for process in processes:
        tagged_process = np.random.poisson(rate, size=len(process))
        tagged_processes.append(tagged_process)
    return tagged_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    # Vectorized process check
    print("Is independent_processes vectorized?", is_vectorized(independent_processes))
    print("Is correlated_processes vectorized?", is_vectorized(correlated_processes))

    # Discretize time and events
    discretized_independent_processes = discretize_time(independent_processes, time)
    discretized_correlated_processes = discretize_time(correlated_processes, time)

    # Sample spikes
    sampled_independent_processes = sample_spikes(independent_processes, time)
    sampled_correlated_processes = sample_spikes(correlated_processes, time)

    # Tag each event with a Poisson process
    tagged_independent_processes = tag_events_with_poisson(independent_processes, rate)
    tagged_correlated_processes = tag_events_with_poisson(correlated_processes, rate)

    # Brand spike times
    brand_spike_times = brand_networks(num_neurons, rate, time)
    brand_spike_times_multiple_rates = brand_networks(num_neurons, rate, time, num_rates=3)


# In[53]:


import numpy as np
from scipy.stats import poisson, binom
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate * b2.Hz) for _ in range(num_neurons * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    processes = np.zeros((num_samples, num_neurons, int(time)))
    spike_indices = np.array([np.array([int(t / b2.ms) for t in spike_train if t < time * b2.ms], dtype=int) for spike_train in spike_trains])
    sample_indices, neuron_indices = np.divmod(np.arange(num_neurons * num_samples), num_neurons)
    processes[sample_indices[:, None], neuron_indices[:, None], spike_indices] = 1  # Set spike times to 1

    # Calculate summing variable, counts, vector set of counts, and characterization of counts
    summing_variable = np.sum(processes)
    counts = np.sum(processes, axis=(0, 2)).flatten()
    vector_set_of_counts = processes.reshape(-1, int(time))
    characterization_of_counts = poisson.pmf(np.sum(processes, axis=(0, 2)), rate * time)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate*b2.Hz) for _ in range((num_neurons + num_common) * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    independent_procs = np.zeros((num_samples, num_neurons + num_common, int(time)), dtype=bool)
    spike_indices = np.array([np.array([int(t / b2.ms) for t in spike_train if t < time * b2.ms], dtype=int) for spike_train in spike_trains])
    sample_indices, neuron_indices = np.divmod(np.arange((num_neurons + num_common) * num_samples), num_neurons + num_common)
    independent_procs[sample_indices[:, None], neuron_indices[:, None], spike_indices] = True

    # Generate correlated processes
    correlated_procs = np.zeros((num_samples, num_neurons, int(time)), dtype=bool)
    correlated_procs[:, :num_common, :] = np.cumsum(independent_procs[:, :num_common, :], axis=2)
    correlated_procs[:, num_common:, :] = independent_procs[:, num_common:, :]

    return correlated_procs

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    # 2D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        plt.plot(range(start, end), processes[0][0][start:end], color=colors[i], label=f'Time Section {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('2D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # 3D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        ax.plot(processes[0][0][start:end], processes[0][1][start:end], np.arange(start, end), color=colors[i], label=f'Time Section {i+1}')
    ax.set_xlabel('Neuron 1 Count')
    ax.set_ylabel('Neuron 2 Count')
    ax.set_zlabel('Time')
    plt.title('3D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # Discrete time
    discrete_time_processes = np.sum(processes, axis=2)
    
    # Continuous time
    continuous_time_processes = np.cumsum(processes, axis=2)
    
    # Plot discrete and continuous time processes
    plt.figure(figsize=(10, 6))
    plt.plot(discrete_time_processes[0, 0], label='Discrete Time')
    plt.plot(continuous_time_processes[0, 0], label='Continuous Time')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Discrete and Continuous Time Poisson Processes')
    plt.legend()
    plt.show()
    
    # Use binomial cross-section for Bernoulli process
    bernoulli_process = np.random.binomial(1, 0.5, size=(len(processes), len(processes[0]), len(processes[0][0])))
    binomial_cross_section = np.sum(bernoulli_process, axis=2)
    plt.figure(figsize=(10, 6))
    plt.plot(binomial_cross_section[0, 0], label='Binomial Cross-Section')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Binomial Cross-Section of Bernoulli Process')
    plt.legend()
    plt.show()

    # Refine code to achieve slowly increasing average

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, rate, time, num_rates=None):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')

    if num_rates is None:
        # Use a single rate for all neurons
        inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    else:
        # Use multiple rates for different neurons
        input_rates = np.random.uniform(0, rate, size=num_rates)
        inputs = []
        for i in range(num_neurons):
            input_rate = input_rates[i % num_rates] * b2.Hz
            input_ = b2.PoissonInput(target=neurons, target_var='v', N=1, rate=input_rate, weight=1)
            inputs.append(input_)

    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def vectorize_code(processes):
    vectorized_processes = []
    for process in processes:
        vectorized_process = np.array(process)
        vectorized_processes.append(vectorized_process)
    return vectorized_processes

def is_vectorized(processes):
    return all(isinstance(process, np.ndarray) for process in processes)

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

def tag_events_with_poisson(processes, rate):
    tagged_processes = []
    for process in processes:
        tagged_process = np.random.poisson(rate, size=len(process))
        tagged_processes.append(tagged_process)
    return tagged_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    # Vectorized process check
    print("Is independent_processes vectorized?", is_vectorized(independent_processes))
    print("Is correlated_processes vectorized?", is_vectorized(correlated_processes))

# Discretize time and events
discretized_independent_processes = discretize_time(independent_processes, time)
discretized_correlated_processes = discretize_time(correlated_processes, time)

print("Discretized independent processes:")
print(discretized_independent_processes[:10])

print("\nDiscretized correlated processes:")
print(discretized_correlated_processes[:10])

# Sample spikes
sampled_independent_processes = sample_spikes(independent_processes, time)
sampled_correlated_processes = sample_spikes(correlated_processes, time)

print("\nSampled independent processes:")
print(sampled_independent_processes[:10])

print("\nSampled correlated processes:")
print(sampled_correlated_processes[:10])

# Tag each event with a Poisson process
tagged_independent_processes = tag_events_with_poisson(independent_processes, rate)
tagged_correlated_processes = tag_events_with_poisson(correlated_processes, rate)

print("\nTagged independent processes:")
print(tagged_independent_processes[:10])

print("\nTagged correlated processes:")
print(tagged_correlated_processes[:10])


# Brand spike times
brand_spike_times = brand_networks(num_neurons, rate, time)
brand_spike_times_multiple_rates = brand_networks(num_neurons, rate, time, num_rates=3)

print("\nBrand spike times:")
for key, value in list(brand_spike_times.items())[:10]:
    print(f"Neuron {key}: {value}")

print("\nBrand spike times multiple rates:")
for key, value in list(brand_spike_times_multiple_rates.items())[:10]:
    print(f"Neuron {key}: {value}")


# In[55]:


import numpy as np
from scipy.stats import poisson, binom
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate * b2.Hz) for _ in range(num_neurons * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    processes = np.zeros((num_samples, num_neurons, int(time)))
    spike_indices = np.array([np.array([int(t / b2.ms) for t in spike_train if t < time * b2.ms], dtype=int) for spike_train in spike_trains])
    sample_indices, neuron_indices = np.divmod(np.arange(num_neurons * num_samples), num_neurons)
    processes[sample_indices[:, None], neuron_indices[:, None], spike_indices] = 1  # Set spike times to 1

    # Calculate summing variable, counts, vector set of counts, and characterization of counts
    summing_variable = np.sum(processes)
    counts = np.sum(processes, axis=(0, 2)).flatten()
    vector_set_of_counts = processes.reshape(-1, int(time))
    characterization_of_counts = poisson.pmf(np.sum(processes, axis=(0, 2)), rate * time)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate*b2.Hz) for _ in range((num_neurons + num_common) * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    independent_procs = np.zeros((num_samples, num_neurons + num_common, int(time)), dtype=bool)
    spike_indices = np.array([np.array([int(t / b2.ms) for t in spike_train if t < time * b2.ms], dtype=int) for spike_train in spike_trains])
    sample_indices, neuron_indices = np.divmod(np.arange((num_neurons + num_common) * num_samples), num_neurons + num_common)
    independent_procs[sample_indices[:, None], neuron_indices[:, None], spike_indices] = True

    # Generate correlated processes
    correlated_procs = np.zeros((num_samples, num_neurons, int(time)), dtype=bool)
    correlated_procs[:, :num_common, :] = np.cumsum(independent_procs[:, :num_common, :], axis=2)
    correlated_procs[:, num_common:, :] = independent_procs[:, num_common:, :]

    return correlated_procs

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes])
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    # 2D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        plt.plot(range(start, end), processes[0][0][start:end], color=colors[i], label=f'Time Section {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('2D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # 3D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        ax.plot(processes[0][0][start:end], processes[0][1][start:end], np.arange(start, end), color=colors[i], label=f'Time Section {i+1}')
    ax.set_xlabel('Neuron 1 Count')
    ax.set_ylabel('Neuron 2 Count')
    ax.set_zlabel('Time')
    plt.title('3D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # Discrete time
    discrete_time_processes = np.sum(processes, axis=2)
    
    # Continuous time
    continuous_time_processes = np.cumsum(processes, axis=2)
    
    # Plot discrete and continuous time processes
    plt.figure(figsize=(10, 6))
    plt.plot(discrete_time_processes[0, 0], label='Discrete Time')
    plt.plot(continuous_time_processes[0, 0], label='Continuous Time')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Discrete and Continuous Time Poisson Processes')
    plt.legend()
    plt.show()
    
    # Use binomial cross-section for Bernoulli process
    bernoulli_process = np.random.binomial(1, 0.5, size=(len(processes), len(processes[0]), len(processes[0][0])))
    binomial_cross_section = np.sum(bernoulli_process, axis=2)
    plt.figure(figsize=(10, 6))
    plt.plot(binomial_cross_section[0, 0], label='Binomial Cross-Section')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Binomial Cross-Section of Bernoulli Process')
    plt.legend()
    plt.show()

    # Refine code to achieve slowly increasing average

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, rate, time, num_rates=None):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')

    if num_rates is None:
        # Use a single rate for all neurons
        inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    else:
        # Use multiple rates for different neurons
        input_rates = np.random.uniform(0, rate, size=num_rates)
        inputs = []
        for i in range(num_neurons):
            input_rate = input_rates[i % num_rates] * b2.Hz
            input_ = b2.PoissonInput(target=neurons, target_var='v', N=1, rate=input_rate, weight=1)
            inputs.append(input_)

    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def is_vectorized(processes):
    return all(isinstance(process, np.ndarray) for process in processes)

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

def tag_events_with_poisson(processes, rate):
    tagged_processes = []
    for process in processes:
        tagged_process = np.random.poisson(rate, size=len(process))
        tagged_processes.append(tagged_process)
    return tagged_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    # Vectorized process check
    print("Is independent_processes vectorized?", is_vectorized(independent_processes))
    print("Is correlated_processes vectorized?", is_vectorized(correlated_processes))

    # Discretize time and events
    discretized_independent_processes = discretize_time(independent_processes[0], time)
    discretized_correlated_processes = discretize_time(correlated_processes[0], time)

    print("Discretized independent processes:")
    print(discretized_independent_processes[:10])

    print("\nDiscretized correlated processes:")
    print(discretized_correlated_processes[:10])

    # Sample spikes
    sampled_independent_processes = sample_spikes(independent_processes[0][0], time)
    sampled_correlated_processes = sample_spikes(correlated_processes[0][0], time)

    print("\nSampled independent processes:")
    print(sampled_independent_processes[:10])

    print("\nSampled correlated processes:")
    print(sampled_correlated_processes[:10])

    # Tag each event with a Poisson process
    tagged_independent_processes = tag_events_with_poisson(independent_processes[0][0], rate)
    tagged_correlated_processes = tag_events_with_poisson(correlated_processes[0][0], rate)

    print("\nTagged independent processes:")
    print(tagged_independent_processes[:10])

    print("\nTagged correlated processes:")
    print(tagged_correlated_processes[:10])

    # Brand spike times
    brand_spike_times = brand_networks(num_neurons, rate, time)
    brand_spike_times_multiple_rates = brand_networks(num_neurons, rate, time, num_rates=3)

    print("\nBrand spike times:")
    for key, value in list(brand_spike_times.items())[:10]:
        print(f"Neuron {key}: {value}")

    print("\nBrand spike times multiple rates:")
    for key, value in list(brand_spike_times_multiple_rates.items())[:10]:
        print(f"Neuron {key}: {value}")


# In[67]:


import numpy as np
from scipy.stats import poisson, binom
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate * b2.Hz) for _ in range(num_neurons * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    processes = np.zeros((num_samples, num_neurons, int(time)))
    spike_indices_list = [[int(t / b2.ms) for t in spike_train if t < time * b2.ms] for spike_train in spike_trains]

    # Pad the lists to have the same length
    max_length = max(len(indices) for indices in spike_indices_list)
    padded_spike_indices_list = [indices + [0] * (max_length - len(indices)) for indices in spike_indices_list]

    # Convert the list of lists to a numpy array
    spike_indices = np.array(padded_spike_indices_list)

    sample_indices, neuron_indices = np.divmod(np.arange(num_neurons * num_samples), num_neurons)
    for i, indices in enumerate(spike_indices):
        processes[sample_indices[i], neuron_indices[i], indices] = 1

    # Calculate summing variable, counts, vector set of counts, and characterization of counts
    summing_variable = np.sum(processes)
    counts = np.sum(processes, axis=(0, 2)).flatten()
    vector_set_of_counts = processes.reshape(-1, int(time))
    characterization_of_counts = poisson.pmf(np.sum(processes, axis=(0, 2)), rate * time)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate*b2.Hz) for _ in range((num_neurons + num_common) * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    independent_procs = np.zeros((num_samples, num_neurons + num_common, int(time)), dtype=bool)
    spike_indices_list = [[int(t / b2.ms) for t in spike_train if t < time * b2.ms] for spike_train in spike_trains]

    # Pad the lists to have the same length
    max_length = max(len(indices) for indices in spike_indices_list)
    padded_spike_indices_list = [indices + [0] * (max_length - len(indices)) for indices in spike_indices_list]

    # Convert the list of lists to a numpy array
    spike_indices = np.array(padded_spike_indices_list)

    sample_indices, neuron_indices = np.divmod(np.arange((num_neurons + num_common) * num_samples), num_neurons + num_common)
    for i, indices in enumerate(spike_indices):
        for index in indices:
            if index < int(time):
                independent_procs[sample_indices[i], neuron_indices[i], index] = True

    # Generate correlated processes
    correlated_procs = np.zeros((num_samples, num_neurons, int(time)), dtype=bool)
    correlated_procs[:, :num_common, :] = np.cumsum(independent_procs[:, :num_common, :], axis=2)
    correlated_procs[:, num_common:, :] = independent_procs[:, num_common:num_neurons, :]

    return correlated_procs


def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes], rowvar=False)
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    # 2D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        plt.plot(range(start, end), processes[0][0][start:end], color=colors[i], label=f'Time Section {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('2D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # 3D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        ax.plot(processes[0][0][start:end], processes[0][1][start:end], np.arange(start, end), color=colors[i], label=f'Time Section {i+1}')
    ax.set_xlabel('Neuron 1 Count')
    ax.set_ylabel('Neuron 2 Count')
    ax.set_zlabel('Time')
    plt.title('3D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # Discrete time
    discrete_time_processes = np.sum(processes, axis=2)
    
    # Continuous time
    continuous_time_processes = np.cumsum(processes, axis=2)
    
    # Plot discrete and continuous time processes
    plt.figure(figsize=(10, 6))
    plt.plot(discrete_time_processes[0, 0], label='Discrete Time')
    plt.plot(continuous_time_processes[0, 0], label='Continuous Time')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Discrete and Continuous Time Poisson Processes')
    plt.legend()
    plt.show()
    
    # Use binomial cross-section for Bernoulli process
    bernoulli_process = np.random.binomial(1, 0.5, size=(len(processes), len(processes[0]), len(processes[0][0])))
    binomial_cross_section = np.sum(bernoulli_process, axis=2)
    plt.figure(figsize=(10, 6))
    plt.plot(binomial_cross_section[0, 0], label='Binomial Cross-Section')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Binomial Cross-Section of Bernoulli Process')
    plt.legend()
    plt.show()

    # Refine code to achieve slowly increasing average

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, firing_rate, time, num_rates=None):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')

    if num_rates is None:
        # Use a single rate for all neurons
        inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=firing_rate*b2.Hz, weight=1)
    else:
        # Use multiple rates for different neurons
        input_rates = np.random.uniform(0, firing_rate, size=num_rates)
        inputs = []
        for i in range(num_neurons):
            input_rate = input_rates[i % num_rates] * b2.Hz
            input_ = b2.PoissonInput(target=neurons, target_var='v', N=1, rate=input_rate, weight=1)
            inputs.append(input_)

    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def is_vectorized(processes):
    return all(isinstance(process, np.ndarray) for process in processes)

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = np.array([process[i] for i in range(0, len(process), resolution)])
        sampled_processes.append(sampled_process)
    return sampled_processes

def tag_events_with_poisson(processes, rate):
    tagged_processes = []
    for process in processes:
        tagged_process = np.random.poisson(rate, size=len(process))
        tagged_processes.append(tagged_process)
    return tagged_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    # Vectorized process check
    print("Is independent_processes vectorized?", is_vectorized(independent_processes))
    print("Is correlated_processes vectorized?", is_vectorized(correlated_processes))

    # Discretize time and events
    discretized_independent_processes = discretize_time(independent_processes[0], time)
    discretized_correlated_processes = discretize_time(correlated_processes[0], time)

    print("Discretized independent processes:")
    print(discretized_independent_processes[:10])

    print("\nDiscretized correlated processes:")
    print(discretized_correlated_processes[:10])

    # Sample spikes
    sampled_independent_processes = sample_spikes(independent_processes[0][0], time)
    sampled_correlated_processes = sample_spikes(correlated_processes[0][0], time)

    print("\nSampled independent processes:")
    print(sampled_independent_processes[:10])

    print("\nSampled correlated processes:")
    print(sampled_correlated_processes[:10])

    # Tag each event with a Poisson process
    tagged_independent_processes = tag_events_with_poisson(independent_processes[0][0], rate)
    tagged_correlated_processes = tag_events_with_poisson(correlated_processes[0][0], rate)

    print("\nTagged independent processes:")
    print(tagged_independent_processes[:10])

    print("\nTagged correlated processes:")
    print(tagged_correlated_processes[:10])

    # Brand spike times
    brand_spike_times = brand_networks(num_neurons, rate, time)
    brand_spike_times_multiple_rates = brand_networks(num_neurons, rate, time, num_rates=3)

    print("\nBrand spike times:")
    for key, value in list(brand_spike_times.items())[:10]:
        print(f"Neuron {key}: {value}")

    print("\nBrand spike times multiple rates:")
    for key, value in list(brand_spike_times_multiple_rates.items())[:10]:
        print(f"Neuron {key}: {value}")


# In[71]:


import numpy as np
from scipy.stats import poisson, binom
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate * b2.Hz) for _ in range(num_neurons * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    processes = np.zeros((num_samples, num_neurons, int(time)))
    spike_indices_list = [[int(t / b2.ms) for t in spike_train if t < time * b2.ms] for spike_train in spike_trains]

    # Pad the lists to have the same length
    max_length = max(len(indices) for indices in spike_indices_list)
    padded_spike_indices_list = [indices + [0] * (max_length - len(indices)) for indices in spike_indices_list]

    # Convert the list of lists to a numpy array
    spike_indices = np.array(padded_spike_indices_list)

    sample_indices, neuron_indices = np.divmod(np.arange(num_neurons * num_samples), num_neurons)
    for i, indices in enumerate(spike_indices):
        processes[sample_indices[i], neuron_indices[i], indices] = 1

    # Calculate summing variable, counts, vector set of counts, and characterization of counts
    summing_variable = np.sum(processes)
    counts = np.sum(processes, axis=(0, 2)).flatten()
    vector_set_of_counts = processes.reshape(-1, int(time))
    characterization_of_counts = poisson.pmf(np.sum(processes, axis=(0, 2)), rate * time)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate*b2.Hz) for _ in range((num_neurons + num_common) * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    independent_procs = np.zeros((num_samples, num_neurons + num_common, int(time)), dtype=bool)
    spike_indices_list = [[int(t / b2.ms) for t in spike_train if t < time * b2.ms] for spike_train in spike_trains]

    # Pad the lists to have the same length
    max_length = max(len(indices) for indices in spike_indices_list)
    padded_spike_indices_list = [indices + [0] * (max_length - len(indices)) for indices in spike_indices_list]

    # Convert the list of lists to a numpy array
    spike_indices = np.array(padded_spike_indices_list)

    sample_indices, neuron_indices = np.divmod(np.arange((num_neurons + num_common) * num_samples), num_neurons + num_common)
    for i, indices in enumerate(spike_indices):
        for index in indices:
            if index < int(time):
                independent_procs[sample_indices[i], neuron_indices[i], index] = True

    # Generate correlated processes
    correlated_procs = np.zeros((num_samples, num_neurons, int(time)), dtype=bool)
    correlated_procs[:, :num_common, :] = np.cumsum(independent_procs[:, :num_common, :], axis=2)
    correlated_procs[:, num_common:, :] = independent_procs[:, num_common:num_neurons, :]

    return correlated_procs

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes], rowvar=False)
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    # 2D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        plt.plot(range(start, end), processes[0][0][start:end], color=colors[i], label=f'Time Section {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('2D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # 3D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        ax.plot(processes[0][0][start:end], processes[0][1][start:end], np.arange(start, end), color=colors[i], label=f'Time Section {i+1}')
    ax.set_xlabel('Neuron 1 Count')
    ax.set_ylabel('Neuron 2 Count')
    ax.set_zlabel('Time')
    plt.title('3D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # Discrete time
    discrete_time_processes = np.sum(processes, axis=2)
    
    # Continuous time
    continuous_time_processes = np.cumsum(processes, axis=2)
    
    # Plot discrete and continuous time processes
    plt.figure(figsize=(10, 6))
    plt.plot(discrete_time_processes[0, 0], label='Discrete Time')
    plt.plot(continuous_time_processes[0, 0], label='Continuous Time')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Discrete and Continuous Time Poisson Processes')
    plt.legend()
    plt.show()
    
    # Use binomial cross-section for Bernoulli process
    bernoulli_process = np.random.binomial(1, 0.5, size=(len(processes), len(processes[0]), len(processes[0][0])))
    binomial_cross_section = np.sum(bernoulli_process, axis=2)
    plt.figure(figsize=(10, 6))
    plt.plot(binomial_cross_section[0, 0], label='Binomial Cross-Section')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Binomial Cross-Section of Bernoulli Process')
    plt.legend()
    plt.show()

# Counting process for two neurons
counting_process_2d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        counting_process_2d[i][j] = count1 + count2

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_2d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Two Neurons')
slope = np.polyfit(range(time), [counting_process_2d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Two Neurons: ", slope)
plt.show()

# Counting process for three neurons
counting_process_3d = [[0 for _ in range(time)] for _ in range(num_samples)]
for i in range(num_samples):
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(time):
        if independent_processes[i][0][j] == 1:
            count1 += 1
        if independent_processes[i][1][j] == 1:
            count2 += 1
        if independent_processes[i][2][j] == 1:
            count3 += 1
        counting_process_3d[i][j] = count1 + count2 + count3

plt.figure(figsize=(10,6))
for j in range(num_samples):
    plt.plot([counting_process_3d[j][i] for i in range(time)])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Counting Process for Three Neurons')
slope = np.polyfit(range(time), [counting_process_3d[0][i] for i in range(time)], 1)[0]
print("Slope of Counting Process for Three Neurons: ", slope)
plt.show()

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def plot_counting_process_2d(processes):
    plt.figure(figsize=(10,6))
    plt.plot(np.cumsum(processes[0][0]), np.cumsum(processes[0][1]))
    plt.xlabel('Count of Neuron 1')
    plt.ylabel('Count of Neuron 2')
    plt.title('2D Plot of Counting Process for Two Neurons')
    plt.show()

def plot_counting_process_3d(processes):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.cumsum(processes[0][0]), np.cumsum(processes[0][1]), np.cumsum(processes[0][2]))
    ax.set_xlabel('Count of Neuron 1')
    ax.set_ylabel('Count of Neuron 2')
    ax.set_zlabel('Count of Neuron 3')
    plt.title('3D Plot of Counting Process for Three Neurons')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, firing_rate, time, num_rates=None):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')

    if num_rates is None:
        # Use a single rate for all neurons
        inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=firing_rate*b2.Hz, weight=1)
    else:
        # Use multiple rates for different neurons
        input_rates = np.random.uniform(0, firing_rate, size=num_rates)
        inputs = []
        for i in range(num_neurons):
            input_rate = input_rates[i % num_rates] * b2.Hz
            input_ = b2.PoissonInput(target=neurons, target_var='v', N=1, rate=input_rate, weight=1)
            inputs.append(input_)

    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def is_vectorized(processes):
    return all(isinstance(process, np.ndarray) for process in processes)

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = process[::resolution]
        sampled_processes.append(sampled_process)
    return sampled_processes

def tag_events_with_poisson(processes, rate):
    tagged_processes = []
    for process in processes:
        tagged_process = np.random.poisson(rate, size=len(process))
        tagged_processes.append(tagged_process)
    return tagged_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
    correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
    network_spike_times = small_network(num_neurons, rate, time)

    means, covariances = calculate_mean_and_covariance(independent_processes)
    slopes = get_slope(independent_processes)

    plot_neurons_accurately(independent_processes, time)
    plot_neuron_against_time(independent_processes, time)
    plot_two_neurons_against_time(independent_processes, time)
    plot_counting_process_2d(independent_processes)
    plot_counting_process_3d(independent_processes)

    print("Independent Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

    print("\nCorrelated Poisson processes:")
    for i in range(num_neurons):
        print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

    print("\nNetwork spike times:")
    for i in range(num_neurons):
        print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

    print("\nSumming variable: {}".format(summing_variable))

    mean_with_std = calculate_mean_with_std(independent_processes)
    print("Mean with standard deviation: ", mean_with_std)

    cov_matrix = calculate_covariance_matrix(independent_processes)
    print("Covariance matrix: ", cov_matrix)

    # Vectorized process check
    print("Is independent_processes vectorized?", is_vectorized(independent_processes))
    print("Is correlated_processes vectorized?", is_vectorized(correlated_processes))

    # Discretize time and events
    discretized_independent_processes = discretize_time(independent_processes[0], time)
    discretized_correlated_processes = discretize_time(correlated_processes[0], time)

    print("Discretized independent processes:")
    print(discretized_independent_processes[:10])

    print("\nDiscretized correlated processes:")
    print(discretized_correlated_processes[:10])

    # Sample spikes
    sampled_independent_processes = sample_spikes(independent_processes[0], time)
    sampled_correlated_processes = sample_spikes(correlated_processes[0], time)
    print("\nSampled independent processes:")
    print(sampled_independent_processes[:10])

    print("\nSampled correlated processes:")
    print(sampled_correlated_processes[:10])

    # Tag each event with a Poisson process
    tagged_independent_processes = tag_events_with_poisson(independent_processes[0][0], rate)
    tagged_correlated_processes = tag_events_with_poisson(correlated_processes[0][0], rate)

    print("\nTagged independent processes:")
    print(tagged_independent_processes[:10])

    print("\nTagged correlated processes:")
    print(tagged_correlated_processes[:10])

    # Brand spike times
    brand_spike_times = brand_networks(num_neurons, rate, time)
    brand_spike_times_multiple_rates = brand_networks(num_neurons, rate, time, num_rates=3)

    print("\nBrand spike times:")
    for key, value in list(brand_spike_times.items())[:10]:
        print(f"Neuron {key}: {value}")

    print("\nBrand spike times multiple rates:")
    for key, value in list(brand_spike_times_multiple_rates.items())[:10]:
        print(f"Neuron {key}: {value}")


# In[73]:


import numpy as np
from scipy.stats import poisson, binom
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt

# Function portion
def independent_poisson_processes(num_neurons: int, rate: float, time: float, num_samples: int) -> tuple:
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate * b2.Hz) for _ in range(num_neurons * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    processes = np.zeros((num_samples, num_neurons, int(time)))
    spike_indices_list = [[int(t / b2.ms) for t in spike_train if t < time * b2.ms] for spike_train in spike_trains]

    # Pad the lists to have the same length
    max_length = max(len(indices) for indices in spike_indices_list)
    padded_spike_indices_list = [indices + [0] * (max_length - len(indices)) for indices in spike_indices_list]

    # Convert the list of lists to a numpy array
    spike_indices = np.array(padded_spike_indices_list)

    sample_indices, neuron_indices = np.divmod(np.arange(num_neurons * num_samples), num_neurons)
    for i, indices in enumerate(spike_indices):
        for index in indices:
            if index < int(time):
                processes[sample_indices[i], neuron_indices[i], index] = 1

    # Calculate summing variable, counts, vector set of counts, and characterization of counts
    summing_variable = np.sum(processes)
    counts = np.sum(processes, axis=(0, 2)).flatten()
    vector_set_of_counts = processes.reshape(-1, int(time))
    characterization_of_counts = poisson.pmf(np.sum(processes, axis=(0, 2)), rate * time)

    return processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts

def correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples):
    # Create a network with multiple Poisson groups
    net = b2.Network()
    poisson_groups = [b2.PoissonGroup(1, rate*b2.Hz) for _ in range((num_neurons + num_common) * num_samples)]
    net.add(poisson_groups)
    spike_monitors = [b2.SpikeMonitor(group) for group in poisson_groups]
    net.add(spike_monitors)
    net.run(time * b2.ms)

    # Extract spike trains and convert to NumPy arrays
    spike_trains = [monitor.spike_trains()[0] for monitor in spike_monitors]
    independent_procs = np.zeros((num_samples, num_neurons + num_common, int(time)), dtype=bool)
    spike_indices_list = [[int(t / b2.ms) for t in spike_train if t < time * b2.ms] for spike_train in spike_trains]

    # Pad the lists to have the same length
    max_length = max(len(indices) for indices in spike_indices_list)
    padded_spike_indices_list = [indices + [0] * (max_length - len(indices)) for indices in spike_indices_list]

    # Convert the list of lists to a numpy array
    spike_indices = np.array(padded_spike_indices_list)

    sample_indices, neuron_indices = np.divmod(np.arange((num_neurons + num_common) * num_samples), num_neurons + num_common)
    for i, indices in enumerate(spike_indices):
        for index in indices:
            if index < int(time):
                independent_procs[sample_indices[i], neuron_indices[i], index] = True

    # Generate correlated processes
    correlated_procs = np.zeros((num_samples, num_neurons, int(time)), dtype=bool)
    correlated_procs[:, :num_common, :] = np.cumsum(independent_procs[:, :num_common, :], axis=2)
    correlated_procs[:, num_common:, :] = independent_procs[:, num_common:num_neurons, :]

    return correlated_procs

def small_network(num_neurons, rate, time):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=rate*b2.Hz, weight=1)
    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def calculate_mean_and_covariance(processes):
    means = []
    covariances = []
    for i in range(len(processes[0])):
        mean = np.mean([process[i] for process in processes])
        covariance = np.cov([process[i] for process in processes], rowvar=False)
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

def get_slope(processes):
    slopes = []
    for i in range(len(processes[0])):
        slope = np.polyfit(range(len(processes[0][i])), processes[0][i], 1)[0]
        slopes.append(slope)
    return slopes

def plot_neurons_accurately(processes, time):
    fig = plt.figure(figsize=(10,6))
    for i in range(len(processes[0])):
        plt.plot(processes[0][i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neurons Over Time')
    plt.legend()
    plt.show()

    # 2D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        plt.plot(range(start, end), processes[0][0][start:end], color=colors[i], label=f'Time Section {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('2D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # 3D Neuron Count vs Time with different colors for sections of time
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    section_size = int(time / 4)  # Divide time into 4 sections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        start = i * section_size
        end = (i + 1) * section_size if i < 3 else time
        ax.plot(processes[0][0][start:end], processes[0][1][start:end], np.arange(start, end), color=colors[i], label=f'Time Section {i+1}')
    ax.set_xlabel('Neuron 1 Count')
    ax.set_ylabel('Neuron 2 Count')
    ax.set_zlabel('Time')
    plt.title('3D Neuron Count vs Time')
    plt.legend()
    plt.show()

    # Discrete time
    discrete_time_processes = np.sum(processes, axis=2)
    
    # Continuous time
    continuous_time_processes = np.cumsum(processes, axis=2)
    
    # Plot discrete and continuous time processes
    plt.figure(figsize=(10, 6))
    plt.plot(discrete_time_processes[0, 0], label='Discrete Time')
    plt.plot(continuous_time_processes[0, 0], label='Continuous Time')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Discrete and Continuous Time Poisson Processes')
    plt.legend()
    plt.show()
    
    # Use binomial cross-section for Bernoulli process
    bernoulli_process = np.random.binomial(1, 0.5, size=(len(processes), len(processes[0]), len(processes[0][0])))
    binomial_cross_section = np.sum(bernoulli_process, axis=2)
    plt.figure(figsize=(10, 6))
    plt.plot(binomial_cross_section[0, 0], label='Binomial Cross-Section')
    plt.xlabel('Time')
    plt.ylabel('Spike Count')
    plt.title('Binomial Cross-Section of Bernoulli Process')
    plt.legend()
    plt.show()

# Counting process for two neurons
def counting_process_2d(processes, time):
    counting_process_2d = np.zeros((len(processes), time))
    for i in range(len(processes)):
        count1 = 0
        count2 = 0
        for j in range(time):
            if processes[i][0][j] == 1:
                count1 += 1
            if processes[i][1][j] == 1:
                count2 += 1
            counting_process_2d[i][j] = count1 + count2
    return counting_process_2d

# Counting process for three neurons
def counting_process_3d(processes, time):
    counting_process_3d = np.zeros((len(processes), time))
    for i in range(len(processes)):
        count1 = 0
        count2 = 0
        count3 = 0
        for j in range(time):
            if processes[i][0][j] == 1:
                count1 += 1
            if processes[i][1][j] == 1:
                count2 += 1
            if processes[i][2][j] == 1:
                count3 += 1
            counting_process_3d[i][j] = count1 + count2 + count3
    return counting_process_3d

def plot_neuron_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    plt.plot(processes[0][0], label='Neuron 1')
    plt.fill_between(range(time), processes[0][0] - np.std(processes[0][0]), processes[0][0] + np.std(processes[0][0]), alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.title('Neuron 1 Over Time with Standard Deviation')
    plt.legend()
    plt.show()

def plot_two_neurons_against_time(processes, time):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processes[0][0], processes[0][1], np.cumsum(processes[0][2]))
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_zlabel('Cumulative Sum of Neuron 3')
    plt.title('Two Neurons Over Time with Cumulative Sum of Third Neuron')
    plt.show()

def plot_counting_process_2d(processes):
    plt.figure(figsize=(10,6))
    plt.plot(np.cumsum(processes[0][0]), np.cumsum(processes[0][1]))
    plt.xlabel('Count of Neuron 1')
    plt.ylabel('Count of Neuron 2')
    plt.title('2D Plot of Counting Process for Two Neurons')
    plt.show()

def plot_counting_process_3d(processes):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.cumsum(processes[0][0]), np.cumsum(processes[0][1]), np.cumsum(processes[0][2]))
    ax.set_xlabel('Count of Neuron 1')
    ax.set_ylabel('Count of Neuron 2')
    ax.set_zlabel('Count of Neuron 3')
    plt.title('3D Plot of Counting Process for Three Neurons')
    plt.show()

def calculate_mean_with_std(processes):
    mean_with_std = np.mean(processes[0][0]) + np.std(processes[0][0])
    return mean_with_std

def calculate_covariance_matrix(processes):
    cov_matrix = np.cov([processes[0][0], processes[0][1]])
    return cov_matrix

def brand_networks(num_neurons, firing_rate, time, num_rates=None):
    neurons = b2.NeuronGroup(num_neurons, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')

    if num_rates is None:
        # Use a single rate for all neurons
        inputs = b2.PoissonInput(target=neurons, target_var='v', N=num_neurons, rate=firing_rate*b2.Hz, weight=1)
    else:
        # Use multiple rates for different neurons
        input_rates = np.random.uniform(0, firing_rate, size=num_rates)
        inputs = []
        for i in range(num_neurons):
            input_rate = input_rates[i % num_rates] * b2.Hz
            input_ = b2.PoissonInput(target=neurons, target_var='v', N=1, rate=input_rate, weight=1)
            inputs.append(input_)

    spike_monitor = b2.SpikeMonitor(neurons)
    net = b2.Network(neurons, inputs, spike_monitor)
    net.run(time * b2.ms)  # Run the network instead of b2.run
    return spike_monitor.spike_trains()

def is_vectorized(processes):
    return all(isinstance(process, np.ndarray) for process in processes)

def discretize_time(processes, time):
    discretized_processes = []
    for process in processes:
        discretized_process = np.array([process[i] for i in range(0, len(process), time)])
        discretized_processes.append(discretized_process)
    return discretized_processes

def sample_spikes(processes, resolution):
    sampled_processes = []
    for process in processes:
        sampled_process = process[::resolution]
        sampled_processes.append(sampled_process)
    return sampled_processes

def tag_events_with_poisson(processes, rate):
    tagged_processes = []
    for process in processes:
        tagged_process = np.random.poisson(rate, size=len(process))
        tagged_processes.append(tagged_process)
    return tagged_processes

# Main portion
if __name__ == "__main__":
    num_neurons = 3
    rate = 10
    time = 1000
    num_common = 2
    num_samples = 100

    if time <= 0 or num_neurons <= 0 or num_samples <= 0:
        print("Error: Time, number of neurons, and number of samples must be positive.")
    else:
        independent_processes, summing_variable, counts, vector_set_of_counts, characterization_of_counts = independent_poisson_processes(num_neurons, rate, time, num_samples)
        correlated_processes = correlated_poisson_processes(num_neurons, rate, time, num_common, num_samples)
        network_spike_times = small_network(num_neurons, rate, time)

        means, covariances = calculate_mean_and_covariance(independent_processes)
        slopes = get_slope(independent_processes)

        plot_neurons_accurately(independent_processes, time)
        plot_neuron_against_time(independent_processes, time)
        plot_two_neurons_against_time(independent_processes, time)
        plot_counting_process_2d(independent_processes)
        plot_counting_process_3d(independent_processes)

        print("Independent Poisson processes:")
        for i in range(num_neurons):
            print("Neuron {}: mean = {}, variance = {}".format(i, sum(independent_processes[0][i]) / len(independent_processes[0][i]), sum([x**2 for x in independent_processes[0][i]]) / len(independent_processes[0][i]) - (sum(independent_processes[0][i]) / len(independent_processes[0][i]))**2))

        print("\nCorrelated Poisson processes:")
        for i in range(num_neurons):
            print("Neuron {}: mean = {}, variance = {}".format(i, sum(correlated_processes[0][i]) / len(correlated_processes[0][i]), sum([x**2 for x in correlated_processes[0][i]]) / len(correlated_processes[0][i]) - (sum(correlated_processes[0][i]) / len(correlated_processes[0][i]))**2))

        print("\nNetwork spike times:")
        for i in range(num_neurons):
            print("Neuron {}: spike times = {}".format(i, network_spike_times.get(i, [])))

        print("\nSumming variable: {}".format(summing_variable))

        mean_with_std = calculate_mean_with_std(independent_processes)
        print("Mean with standard deviation: ", mean_with_std)

        cov_matrix = calculate_covariance_matrix(independent_processes)
        print("Covariance matrix: ", cov_matrix)

        # Vectorized process check
        print("Is independent_processes vectorized?", is_vectorized(independent_processes))
        print("Is correlated_processes vectorized?", is_vectorized(correlated_processes))

        # Discretize time and events
        discretized_independent_processes = discretize_time(independent_processes[0], time)
        discretized_correlated_processes = discretize_time(correlated_processes[0], time)

        print("Discretized independent processes:")
        print(discretized_independent_processes[:10])

        print("\nDiscretized correlated processes:")
        print(discretized_correlated_processes[:10])

        # Sample spikes
        sampled_independent_processes = sample_spikes(independent_processes[0], time)
        sampled_correlated_processes = sample_spikes(correlated_processes[0], time)
        print("\nSampled independent processes:")
        print(sampled_independent_processes[:10])

        print("\nSampled correlated processes:")
        print(sampled_correlated_processes[:10])

        # Tag each event with a Poisson process
        tagged_independent_processes = tag_events_with_poisson(independent_processes[0][0], rate)
        tagged_correlated_processes = tag_events_with_poisson(correlated_processes[0][0], rate)

        print("\nTagged independent processes:")
        print(tagged_independent_processes[:10])

        print("\nTagged correlated processes:")
        print(tagged_correlated_processes[:10])

        # Brand spike times
        brand_spike_times = brand_networks(num_neurons, rate, time)
        brand_spike_times_multiple_rates = brand_networks(num_neurons, rate, time, num_rates=3)

        print("\nBrand spike times:")
        for key, value in list(brand_spike_times.items())[:10]:
            print(f"Neuron {key}: {value}")

        print("\nBrand spike times multiple rates:")
        for key, value in list(brand_spike_times_multiple_rates.items())[:10]:
            print(f"Neuron {key}: {value}")


# In[ ]:





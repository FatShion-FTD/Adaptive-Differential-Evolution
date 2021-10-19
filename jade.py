import os
import glob
from math import fabs, sqrt
import time
import numpy as np
from halton import halton
import random


# JADE
def jade(thetas_limit, target_vectors_init, uCR=0.5, uF=0.6, c=0.1, NP=10, max_gen=100, experiment_name='jade'):
    # n_params = len(thetas_limit)
    n_params = thetas_limit
    # Generate random target vectors
    # target_vectors = np.random.rand(NP, n_params)
    target_vectors = target_vectors_init.copy()
    target_fitness = np.zeros(NP)
    # target_vectors = np.interp(target_vectors, (0,1), (-1,1))

    halton_vectors = halton(NP+1, 265).evaluate()[1:]
    notimprov = 0
    prev_best = 0
    # Initial evaluate target vectors
    for pop in range(NP):
        target_fitness[pop] = simulation(target_vectors[pop])

    # Variable donor vectors
    donor_vector = np.zeros(n_params)
    # Variable trial vectors
    trial_vector = np.zeros(n_params)
    Fi = np.zeros(NP)
    CRi = np.zeros(NP)
    onethirdNP = NP//3
    for gen in range(max_gen):
        print("Generation :", gen)
        # Success F and Success CR
        sCR = []
        sF = []
        random_onethird_idx = np.random.choice(np.arange(0,NP), size=onethirdNP, replace=False).tolist()
        # Generate adaptive parameter Fi and CRi
        for pop in range(NP):
            CRi[pop] = np.random.normal(uCR, 0.1)
            if pop in random_onethird_idx:
                Fi[pop] = np.interp(np.random.rand(), (0,1), (0,1.2))
            else:
                Fi[pop] = np.random.normal(uF, 0.1)


        for pop in range(NP):

            # current_best_idx = np.argmin(target_fitness) 
            current_best_idx = np.argmax(target_fitness)

            index_choice = [i for i in range(NP) if i != pop]
            a, b = np.random.choice(index_choice, 2)
            # calculate JAD function
            donor_vector = target_vectors[pop] + Fi[pop] * (target_vectors[current_best_idx]-target_vectors[pop]) + Fi[pop] * (target_vectors[a]-target_vectors[b])

            # check bondary
            for t in range(n_params):
                if(donor_vector[t] > 1 or donor_vector[t] < -1):
                    donor_vector[t] = random.uniform(-1, 1)

            cross_points = np.random.rand(n_params) <= CRi[pop]
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])

            
            trial_fitness = simulation(trial_vector)
            
            if trial_fitness > target_fitness[pop]:
                target_vectors[pop] = trial_vector.copy()
                target_fitness[pop] = trial_fitness
                sCR.append(CRi[pop])
                sF.append(Fi[pop])
        

        uCR = (1-c)*uCR + c*np.mean(sCR)
        uF = (1-c)*uF + c*(np.sum(np.power(sF, 2))/np.sum(sF))

        # Save and calculate results
        cur_round = gen + 1
        best = np.argmax(target_fitness)
        std = np.std(target_fitness)
        mean = np.mean(target_fitness)
        # saves results
        file_aux = open(experiment_name+'/results.txt', 'a')
        print('\n GENERATION '+str(cur_round)+' '+str(round(target_fitness[best], 6))
              + ' '+str(round(mean, 6))+' '+str(round(std, 6)))
        file_aux.write('\n'+str(cur_round)+' '+str(round(target_fitness[best], 6))+' '
                       + str(round(mean, 6))+' '+str(round(std, 6)))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt', target_vectors[best])

        # saves simulation state
        solutions = [target_vectors, target_fitness]
        env.update_solutions(solutions)
        env.save_state()

        # long time no improvemnt, using halton seq
        if prev_best == target_fitness[best]: 
            notimprov += 1
        else:   notimprov = 0
        prev_best = target_fitness[best]
        
        if notimprov >= 5:
            # halton shaking
            for i in range(0,5): # repeat 5 times
                rand_vector_index = np.random.randint(low=0, high=NP)
                # cannot touch best
                while rand_vector_index == best: rand_vector_index = np.random.randint(low=0, high=NP)
                target_vectors[rand_vector_index] = target_vectors[rand_vector_index] * (1 + 0.5 * halton_vectors[rand_vector_index])
                for t in range(n_params):
                    if(target_vectors[rand_vector_index][t] > 1 or target_vectors[rand_vector_index][t] < -1):
                        target_vectors[rand_vector_index][t] = random.uniform(-1, 1)

            

# runs simulation
def simulation(x):
    # your simulation function for each indivudal
    return 0



experiment_name = 'jade'          
npop = 20   # population size
gens = 150  # generations 
n_vars = 20 # dimensions

ini = time.time()  # sets time marker

# evolution
jade(n_vars, pop, NP=npop, experiment_name=experiment_name,max_gen=gens)

fim = time.time()  # prints total execution time for experiment

print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

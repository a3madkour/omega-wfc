from omega.symbolic.fol import Context
import json
import time
import random
import uuid
import os
# from draw import draw_sample
    

from SampleBDD import SampleBDD
from Experiment import Experiment, Metric
from utils import get_subdirectory, get_all_elements_in_dir
       


def gen_n_models(generator,n,check_sat=False):
    models = []
    it = generator.bdd.pick_iter(generator.bdd_node)
    total_number_of_models =  int(generator.bdd.count(generator.bdd_node))
    actual_number_of_models = n
    if n > total_number_of_models:
        print(f"In gen_n_models specified n,{n},is greater than the number of models in the generator,{total_number_of_models},so we are clipping")
        actual_number_of_models = total_number_of_models


    for i in range(actual_number_of_models):
        model = next(it)
        model_bit_string = generator.sample_as_bit_string(model)
        if check_sat:
            if not generator.sat_omega_model(model):
                print(f"yo we got an unsat model over here: {model}")
                
        models.append((model,model_bit_string))

    return (models,actual_number_of_models)



def save_models(filename,models, append=False):
    if append:
        f = open(f"{filename}", 'a')
    else:
        f = open(f"{filename}", 'w')
    for model_tup in models:
        (_, model_bit_string) = model_tup
        f.write(f"{model_bit_string}\n")
    f.close()

# names =  ["Castle", "Circles","Circuit", "FloorPlan", "Knots", "Rooms", "Summer"]
names = ["FloorPlan", "Rooms"]
# sizes = [2,5]
sizes = [2,5,10]
num_models = [10,100,1000,10000,100000]
for name in names:
    for dim in sizes:
        for num_mod in num_models:
            experiment = Experiment(dim=dim)
            # print(experiment)
            # experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
            experiment.run(num_samples=1)
            (models,actual_number_of_models) = gen_n_models(experiment.generator,num_mod)
            path = get_subdirectory(f"{name}-{dim}x{dim}")
            save_models(f"{path}/first_{actual_number_of_models}_models.txt",models)
        # experiment.to_csv("back")
        # print(experiment)
        # print(experiment)
        # generator = experiment.generator
        # (_,_,sample) = generator.sample()
        # for i in range(10):
        #     (_,_,sample) = generator.sample()
        #     is_true = generator.sat_sample(sample)
        #     # print(len(sample))
        #     # if not is_true:
        #     #     print("not valid")
            
        #     assignment = generator.get_assignment(sample)
        #     final_img = generator.draw_simple_tiled(assignment)
        #     final_img.save(f"temp{i}.png")
        # experiment.generator.dump_bdd("temp.json")
        # picked_model = generator.bdd.pick(generator.bdd_node)
        # models = gen_n_models(generator,10000)
        # print(models)
        # generator.gen_uniform_training_set("train", generator.tile_vec,N,generator.tile_size, 10)

        


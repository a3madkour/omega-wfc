from omega.symbolic.fol import Context, _refine_assignment
import json
import time
import random
import uuid
import os
# from draw import draw_sample
    

from SampleBDD import SampleBDD
from Experiment import Experiment, Metric
from utils import get_subdirectory, get_all_elements_in_dir
from SimpleTiled import SimpleTiled, TileSet
from Platformer import Platformer
import numpy as np
       

# tileset_path = "TheVGLC/Super Mario Bros/smb.json" 
# json_file = open(tileset_path, 'r')
# load_json = json.load(json_file)
# print(load_json)
# names =  ["Castle", "Circles","Circuit", "FloorPlan", "Knots", "Rooms", "Summer"]
# names = ["FloorPlan", "Rooms"]
# sizes = [2,5]
# sizes = [2,5,10]
# num_models = [10,100,1000,10000,100000]

# for line in f.readlines():
#     experiment = Experiment(dim=2)
#     experiment.run()
#     models = experiment.generator.gen_n_models(1)

#     break



# width = 10
# height = 4
# generator = Platformer(width = width, height=height)
# generator.compile()
# spec = generator.bdd_node
# context = generator.context
# reachable = generator.reachable
# print(dir(spec))
# assigns = [assign]
#as from takes you from assignment to variables


# (_,_,sample,marginal) = generator.sample()
# print(generator.tree_true_probs )
# print(marginal)
# print(f"reachable: {reachable}")

# print(reachable)
design = np.zeros((2,2),dtype=float)
# print(design)
# sol = context.pick(spec)
# sol = spec.pick()

# bit_string = generator.sample_as_bit_map(sol)
# print(generator.assignment_from_bit_string(bit_string))
        # temp_design.append()
# concrete_design = context.assign_from(sol)

# design = np.zeros((width,height),dtype=str)
# print(sol)
# for (i,j),name in generator.var_names.items():
#     if name in sol and sol[name]:
#         design[i,j] = "X"
#     else:
#         design[i,j] = "-"

# print(design)

# print(concrete_design)
# design = []
# for i in range(width):
#     temp_design = []
#     for j in range(height):
#         #context being false means it is not solid
#         if reachable[(i,j)] & concrete_design != context.false:
#             temp_design.append("-")

            
# for i,j in reachable:
#   if reachable[(i,j)] & concrete_design != context.false:
#     design[i,j] = min(1,design[i,j]+0.2)

# for i,j in reachable:
#     print("asda", i,j)
#     if reachable[(i,j)]: 
#         design[i,j] = min(1,design[i,j]+0.2)

# print(design)

# print(spec.bdd.cube([assign]))
# print(generator.bdd.vars)
# for i in range(3):
#   sample_spec = spec
  # pins = np.zeros((generator.width,generator.height),dtype=bool)
  # print(pin)


# generator.gen_sample_training_set("training",num_samples=10)
# # experiment = Experiment(generator = generator, dim=2, num_samples = 10)
# experiment = Experiment(dim=2, num_samples = 10)
# experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
# experiment.run(num_trials=1)
# experiment.to_csv(f"experiments")
num_iterations = 1000
num_samples = [1000,10000]
dims = [2,5,10]
for i in range(num_iterations):
    for num_sample in num_samples:
        for dim in dims:
            experiment = Experiment(dim = dim, num_samples=num_sample)
            experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
            experiment.to_csv(f"Knots-{num_samples}")

# dims = [2]
# epochs = [100,1000,10000]
# for num_samples in num_samples:
#     for dim in dims:
#         experiment = Experiment(dim=dim, num_samples = num_samples)
#         experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
#         experiment.run(num_trials=1)
#         experiment.to_csv(f"no-learning-{num_samples}-samples")
#         for epoch in epochs:
#             (actual_number_of_models,models) = experiment.generator.gen_n_models(epoch)
#             actual_models = []
#             for model in models:
#                 (actual_model, model_string) = model
#                 actual_models.append(actual_model)
#             assignments = []
#             random.shuffle(actual_models)
#             experiment.generator.assign_weights()
#             experiment.generator.train_with_one_run(actual_models)
#             experiment.run(clear_true_probs=True, num_trials=1)
#             experiment.to_csv(f"train-on-{actual_number_of_models}-models-{num_samples}-samples")

# for name in names:
#     for dim in sizes:
#         for num_mod in num_models:
#             experiment = Experiment(dim=dim)
#             # print(experiment)
#             # experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
#             experiment.run(num_samples=1)
#             (models,actual_number_of_models) = gen_n_models(experiment.generator,num_mod)
#             path = get_subdirectory(f"{name}-{dim}x{dim}")
#             save_models(f"{path}/first_{actual_number_of_models}_models.txt",models)
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

        


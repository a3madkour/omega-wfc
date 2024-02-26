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
       
# names =  ["Castle", "Circles","Circuit", "FloorPlan", "Knots", "Rooms", "Summer"]
# names = ["FloorPlan", "Rooms"]
# sizes = [2,5]
# sizes = [2,5,10]
# num_models = [10,100,1000,10000,100000]

f = open("models/Simpletiled/Knots-2x2/first_10_models.txt", "r")
# for line in f.readlines():
#     experiment = Experiment(dim=2)
#     experiment.run()
#     models = experiment.generator.gen_n_models(1)

#     break



num_samples = [1000,10000]
dims = [2]
epochs = [100,1000,10000]
for num_samples in num_samples:
    for dim in dims:
        experiment = Experiment(dim=dim, num_samples = num_samples)
        experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
        experiment.run(num_trials=1)
        experiment.to_csv(f"no-learning-{num_samples}-samples")
        for epoch in epochs:
            (actual_number_of_models,models) = experiment.generator.gen_n_models(epoch)
            actual_models = []
            for model in models:
                (actual_model, model_string) = model
                actual_models.append(actual_model)
            assignments = []
            experiment.generator.assign_weights()
            experiment.generator.train_with_one_run(actual_models)
            experiment.run(clear_true_probs=True, num_trials=1)
            experiment.to_csv(f"train-on-{actual_number_of_models}-models-{num_samples}-samples")

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

        


from omega.symbolic.fol import Context
import json
import time
import random
import uuid
import os
# from draw import draw_sample
    

from SampleBDD import SampleBDD
from Experiment import Experiment
from utils import get_subdirectory, get_all_elements_in_dir
       


# names =  ["Castle", "Circles","Circuit", "FloorPlan", "Knots", "Rooms", "Summer"]
names = ["Knots"]
# sizes = [2,5]
sizes = [2]
for name in names:
    for N in sizes:
        experiment = Experiment(dim=N)
        experiment.run()
        # print(experiment)
        # bdd_wrapper = SampleBDD(generator)
        # sample = bdd_wrapper.sample()
        # bdd_wrapper.dump_bdd("temp.json")
        # bdd_wrapper.gen_uniform_training_set("train", tile_vec,N,tile_size, 10)
        # bdd_wrapper.train_with_n_runs("train")
        # assignment = bdd_wrapper.get_assignment(sample,tile_vec,N,tile_size)
        # final_img = draw_simple_tiled(assignment,tile_vec,N,tile_size)
        # final_img.save("temp.png")

        


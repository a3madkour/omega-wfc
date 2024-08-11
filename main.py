from Experiment import Experiment, draw_simple_tiled_heat_map,Metric, draw_analytical_era_simpletiled
from Experiment import asp_run_test,draw_simple_tiled_samples,draw_asp_tilemap
from SimpleTiled import SimpleTiled,TileSet
from Platformer import Platformer, draw_analytical_era_reachable_platformer
from Maze import Maze
import numpy as np
from utils import props_from_graph, maze_props, sol_props 
import matplotlib.pyplot as plt

from utils import get_subdirectory, get_all_elements_in_dir,props_from_graph, maze_props, sol_props 






# draw_asp_tilemap("Knots", 2, 1)

width = 10
height = 8

# generator = Platformer(width=width, height=height)

generator = Maze(dim=5)
# print("compiling")
compile_time = generator.compile()
# print(f"Done: Compiling {compile_time / 1e09}s")
# print(generator.bdd_node.pick())
# print("done compiling")

# file_name = generator.gen_sample_training_set("training")
# dir_path = get_subdirectory("training")
# assignments = generator.load_sample_assignments_set(f"{dir_path}/{file_name}")
# counts = generator.train_with_one_run(assignments,False)
# print(counts)
# print("Assignements:", assignments)

# print(generator.bdd_node.count())

# (
#     compute_true_probs_time,
#     sample_time,
#     final_sample,
#     sample_marginal
# ) = generator.sample(clear_true_probs=False)

# print(final_sample)

# assignment = generator.get_assignment(final_sample)
# graph= generator.get_maze_graph(assignment)
# cell_type = props_from_graph(graph,generator.dim)
# maze_prop = maze_props(cell_type)
# sol_prop = sol_props(graph,generator.dim, cell_type)
# print(sol_prop)
# print(maze_prop)

generator.gen_sample_training_set("train-maze",10)
path = "train-maze"
runs = get_all_elements_in_dir(get_subdirectory(path), lambda x: "pkl" in x)
    
# print(runs)


assignments = generator.load_sample_assignments_set(path+"/"+runs[0])
print("before training: ", len(assignments))
# generator.train_with_n_runs(path)
# selected_edge = generator.var_names[(0,1)]
# print(generator.weights[selected_edge])
# print(generator.weights)

num_runs = 10
num_pass_trained = []
num_pass_random = []
assign_sets = set()
# for n in range(num_runs):
#     assignments = []
#     for i in range(1000):
#         (
#             compute_true_probs_time,
#             sample_time,
#             final_sample,
#             sample_marginal
#         ) = generator.sample(clear_true_probs=False, useWeights=True)
#         # print(generator.weights)

#         assignment = generator.get_assignment(final_sample)
#         graph= generator.get_maze_graph(assignment)
#         cell_type = props_from_graph(graph,generator.dim)
#         maze_prop = maze_props(cell_type)
#         sol_prop = sol_props(graph,generator.dim, cell_type)
#         selected_edge = generator.var_names[(0,1)]
#         if 'tu' in sol_prop:
#             if sol_prop['tu'] >= 5:
#                 assignments.append(assignment)

#     num_pass = len(assignments)
#     num_pass_trained.append(num_pass)

# # for bdd in generator.tree_true_probs:
# #     print(f"var:{generator.hash_to_var[bdd]}, prob:{generator.tree_true_probs[bdd]}")
# #     if generator.hash_to_var[bdd] == "edge0_1":
# #         print(generator.tree_true_probs[bdd])

# generator.tree_true_probs.clear()
# generator.weights.clear()
# for n in range(num_runs):
#     assignments = []
#     for i in range(1000):
#         (
#             compute_true_probs_time,
#             sample_time,
#             final_sample,
#             sample_marginal
#         ) = generator.sample(clear_true_probs=False, useWeights=False)
#         # print(generator.weights)

#         assignment = generator.get_assignment(final_sample)
#         graph= generator.get_maze_graph(assignment)
#         cell_type = props_from_graph(graph,generator.dim)
#         maze_prop = maze_props(cell_type)
#         sol_prop = sol_props(graph,generator.dim, cell_type)
#         # selected_edge = generator.var_names[(0,1)]
#         if 'tu' in sol_prop:
#             if sol_prop['tu'] >= 5:
#                 assignments.append(assignment)

#     num_pass = len(assignments)
#     num_pass_random.append(num_pass)


# print(np.average(num_pass_random), np.std(num_pass_random))
# print(np.average(num_pass_trained), np.std(num_pass_trained))



# generator.draw_maze(graph)
# print(assignment)
# solid = []
# for i in range(width):
    # solid.append([])
    # solid[i] = []
    # for j in range(height):
    #   solid[i].append(0)
# for assign in assignement:
#     els = assign.split("_")
    # x = int( els[1] )
    # y = int( els[2] )
    # val = int(assignement[assign])
    #doing height - y because y-axis is reversed
#     solid[x][y] = val
# sl = np.array(solid).T
# plt.imshow(sl, interpolation="none")
# plt.savefig("foo.png")

# draw_analytical_era_reachable_platformer(generator,False)
# bmp = generator.sample_as_bit_map(final_sample)
# # assignment = generator.get_assignment(bmp)
# print(assignment)

# for i in range(10):
#     generator
#     (
#         compute_true_probs_time,
#         sample_time,
#         final_sample,
#         sample_marginal
#     ) = generator.sample(clear_true_probs=False)
#     bmp = generator.sample_as_bit_map(final_sample)
#     # print("bmp: ", bmp)
#     assignment = generator.get_assignment(bmp)
#     # print(assignment)
#     img = generator.draw_simple_tiled(assignment)
#     img.save(f"samples/example-sample{i}.png")
# #
# generator.

# experiment = Experiment()
# experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
# experiment.run(num_trials=1, num_samples = 1)
# draw_simple_tiled_samples(experiment)

# draw_simple_tiled_heat_map(experiment)
# draw_analytical_era_simpletiled(experiment)

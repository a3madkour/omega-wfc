from omega.symbolic.fol import Context, _refine_assignment
import json
from PIL import Image, ImageOps
import time
import subprocess
import random
import uuid
import pandas as pd
import os
import timeit

# from draw import draw_sample


from SampleBDD import SampleBDD
from Experiment import Experiment, Metric
from utils import get_subdirectory, get_all_elements_in_dir
from SimpleTiled import SimpleTiled, TileSet
from Platformer import Platformer

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from ASP import facts_from_tileset, asp_facts_from_tiles


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


def run_test(filename, output_file, ground_time, dim, sample_num):
    #this is dumb
    generator = SimpleTiled(dim=2)
    generator.compile()

    start = time.time()
    seed = int(random.random() * 10)
    # cmd_str = (
    #     "clingo wfc.lp --seed="
    #     + str(seed)
    #     + " "
    #     + filename
    #     + "-grounded.lp -n 1 --outf=2 --sign-def=rnd --rand-freq=1"
    # )
    cmd_str = (
        f"clingo --seed={seed} platformer-grounded.lp -n 1 --outf=2 --sign-def=rnd --rand-freq=1"
    )
    print(cmd_str)
    res = json.loads(subprocess.run(cmd_str, shell=True, capture_output=True).stdout)
    end = time.time()
    solve_time = res["Time"]["Total"] * 1e9
    # if solve_time == 0:
    #     solve_time = (end - start)* 1e+9
    # print("solve: ",res["Time"]["Solve"] * 1e+9)
    print((end - start) * 1e9)
    atoms = res["Call"][-1]["Witnesses"][-1]["Value"]
    print(atoms)
    for atom in atoms:
        if atom.startswith("shape("):
            shape = eval(atom[len("shape") :])


    print(atoms)
    assignment = {}
    # result_tiles = np.zeros(shape)
    hash_str = ""
    beep = []
    for i in range(10):
        boop = []
        for j in range(8):
            boop.append(0)
        beep.append(boop)
            
    for atom in atoms:
        if atom.startswith("solid("):
            (i, j) = eval(atom[len("solid") :])
            beep[i-1][j-1] = 1
            # result_tiles[i, j] = v
            
    for boop in beep:
        for bit in boop:
            hash_str = f"{hash_str}{bit}"

    print("assignment: ", assignment)
    # im = generator.draw_simple_tiled_asp(assignment)
    # image_name = f"asp-imgs/{filename}-{sample_num}.png"
    # im = ImageOps.mirror(im)
    # im.save(image_name)

    output_file.write(
        "ASP," + filename + "," + str(solve_time) + "," + str(ground_time) + "," + hash_str +"\n"
    )
    # return result_tiles


def draw_asp_tilemap(name="Knots", dim=2,num_samples = 1000):
    f = open("asp.csv", "a")
    gen = facts_from_tileset(name, asp_facts_from_tiles, dim)
    samples = []
    counts = []
    for i in range(dim):
        rows = []
        for j in range(dim):
            columns = []
            for k in range(13):
                columns.append(0)
                rows.append(columns)
        counts.append(rows)

    for k in range(num_samples):
        start = time.time()
        # res = subprocess.run(
        #     f"time gringo wfc.lp {name}.lp > {name}-grounded.lp",
        #     shell=True,
        #     capture_output=True,
        # )
        res = subprocess.run(
            f"time gringo platformer.lp > platformer-grounded.lp",
            shell=True,
            capture_output=True,
        )
        end = time.time()
        ground_time = (end - start) * 1e9
        sample = np.array(run_test(name, f, ground_time, dim,k))

    #     for i in range(dim):
    #         for j in range(dim):
    #             cell_value = sample[i][j]
    #             print(cell_value)
    #             counts[i][j][int(cell_value)] += 1

    # count_np = np.array(counts)
    # fig, axes = plt.subplots(2, 2)
    # for i, row in enumerate(axes):
    #     for j, ax in enumerate(row):
    #         x_labels = [f"{i}" for i in range(len(counts[i][j]))]
    #         data = np.array(counts[i][j])
    #         # data = data / experiment.num_samples
    #         heatmap = sb.barplot(x=x_labels, y=data, ax=ax, linewidth=1)
    #         heatmap.set_xticks(
    #             ticks=[i for i in range(len(counts[i][j]))],
    #             labels=x_labels,
    #             rotation=0,
    #         )
    #         heatmap.set_ylim(1,400)

    #         # fig.set_ylabel("Count")
    #         fig.set_figheight(15)
    #         fig.set_figwidth(15)
    #         fig.savefig("asp_tile_set.png")



draw_asp_tilemap(dim=10, num_samples=1000)
# width = 10
# height = 4
# generator = Platformer(width=width, height=height)


# delta of python code is neg in this case
# ground_time = res["Time"]["Total"] * 1e+9

# print(np.unique(samples, return_counts=True, axis=1))

# t = timeit.Timer(subprocess.run(f"gringo wfc.lp {name}.lp > {name}-grounded.lp", shell=True, capture_output=True))
# print(df)
# generator.compile()
# spec = generator.bdd_node
# context = generator.context
# reachable = generator.reachable
# print(dir(spec))
# assigns = [assign]
# as from takes you from assignment to variables


# (_,_,sample,marginal) = generator.sample()
# print(generator.tree_true_probs )
# print(marginal)
# print(f"reachable: {reachable}")

# print(reachable)
# design = np.zeros((2, 2), dtype=float)
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


# TODO: plot the assignment simple_tiled samples
def draw_simple_tiled_samples(experiment):
    generator = experiment.generator
    for trial in experiment.trials:
        for bit_string in trial.metrics_data[metric]:
            assignment = generator.assignment_from_bit_string(bit_string)
            index_assignment = generator.index_assignment_from_bit_string(bit_string)
            tile_num_assign = generator.get_assignment(index_assignment)
            # tile_assignment =
            final_img = generator.draw_simple_tiled(tile_num_assign)
            final_img.save(f"sample_images/{generator.spec_string()}/{bit_string}.png")
            # print(bit_string)
            # print(assignment)
            # TODO: heatmap of variable assignment
            # TODO: analytic ERA via model counts
    #


# TODO: Heatmap of variable assignment


# make it ourselves


def draw_simple_tiled_heat_map(experiment, prob=False):
    generator = experiment.generator
    for trial in experiment.trials:
        heat_map_arr = []
        counts = []
        for i in range(generator.dim):
            rows = []
            for j in range(generator.dim):
                columns = []
                print(dir(generator.context))
                for k in range(len(generator.tile_vec)):
                    columns.append(0)
                rows.append(columns)
            counts.append(rows)

        for bit_string in trial.metrics_data[metric]:
            assignment = generator.assignment_from_bit_string(bit_string)
            index_assignment = generator.index_assignment_from_bit_string(bit_string)
            tile_num_assign = generator.get_assignment(index_assignment)

            # print(counts)
            for i, assign in enumerate(tile_num_assign):
                for j in range(len(assign)):
                    counts[i][j][assign[j]] += 1

        count_np = np.array(counts)
        fig, axes = plt.subplots(generator.dim, generator.dim)
        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                x_labels = [f"{i}" for i in range(len(counts[i][j]))]
                data = np.array(counts[i][j])
                if prob:
                    data = data / experiment.num_samples

                heatmap = sb.barplot(x=x_labels, y=data, ax=ax, linewidth=1)
                # bbox = ax.get_tightbbox(fig.canvas.get_renderer())
                # x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
                # xpad = 0.01 * width
                # ypad = 0.01 * height
                # fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+xpad, height+ypad, edgecolor='black', linewidth=3, fill=False))
                heatmap.set_xticks(
                    ticks=[i for i in range(len(counts[i][j]))],
                    labels=x_labels,
                    rotation=0,
                )
                heatmap.set(ylim=(1,400))

        # fig.set_ylabel("Count")
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.savefig("wfc_era.png")


def draw_analytical_era_simpletiled(generator, prob=False):
    counts = []
    for i in range(generator.dim):
        rows = []
        for j in range(generator.dim):
            columns = []
            index = i * generator.dim + j
            var_name = generator.varnames[(i, j)]
            for k in range(len(generator.tile_vec)):
                new_bdd = generator.context.let({var_name: k}, generator.bdd_node)
                columns.append(new_bdd.count())
            rows.append(columns)
        counts.append(rows)

    count_np = np.array(counts)
    print(count_np.shape)
    fig, axes = plt.subplots(generator.dim, generator.dim)
    print(count_np)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            x_labels = [f"{i}" for i in range(len(counts[i][j]))]
            data = np.array(counts[i][j])
            if prob:
                data = data / experiment.num_samples

            heatmap = sb.barplot(x=x_labels, y=data, ax=ax, linewidth=1)
            # bbox = ax.get_tightbbox(fig.canvas.get_renderer())
            # x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
            # xpad = 0.01 * width
            # ypad = 0.01 * height
            # fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+xpad, height+ypad, edgecolor='black', linewidth=3, fill=False))
            heatmap.set_xticks(
                ticks=[i for i in range(len(counts[i][j]))],
                labels=x_labels,
                rotation=0,
            )
            heatmap.set(ylim=(1,400))

    # fig.set_ylabel("Count")
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.savefig("wfc_analytical_era.png")

    # print(counts)
    # for bit_string in trial.metrics_data[metric]:
    # assignment = generator.assignment_from_bit_string(bit_string)
    # index_assignment = generator.index_assignment_from_bit_string(bit_string)
    # tile_num_assign = generator.get_assignment(index_assignment)

    # # print(counts)
    # for i, assign in enumerate(tile_num_assign):
    #     for j in range(len(assign)):
    # counts[i][j][assign[j]] += 1


def draw_analytical_era_platformer(generator, prob=False):
    counts = generator.analytical_expressive_range_analysis()
    grid_of_counts = []
    for i in range(generator.width):
        solid_counts = []
        for j in range(generator.height):
            index = i * generator.height + j
            var_at_level = generator.bdd.var_at_level(index)

            count = counts[var_at_level][0]
            if prob:
                count = count / generator.bdd_node.count()

            solid_counts.append(count)
        grid_of_counts.append(solid_counts)

    # print(grid_of_counts)
    df = pd.DataFrame(grid_of_counts)
    heatmap = sb.heatmap(df.transpose(), cmap="crest", linewidth=1, square=True)
    heatmap.set(xticklabels=[])
    heatmap.set(yticklabels=[])
    heatmap.tick_params(bottom=False, left=False)
    fig = heatmap.get_figure()
    fig.savefig("platformer_analytical_prob.png")
    fig.clf()


def draw_analytical_era_platformer_given_path():
    pass


def draw_analytical_era_reachable_platformer(generator, prob=False):
    # countable models where a given tile is reachable
    grid_of_counts = []
    for i in range(generator.width):
        solid_counts = []
        for j in range(generator.height):
            reach_bdd_node = generator.reachable[(i, j)]
            actual_node_to_count = reach_bdd_node & generator.bdd_node
            count = actual_node_to_count.count()

            if prob:
                count = count / generator.bdd_node.count()
                if count > 1:
                    print("how")
                    print(f"i,j: {i},{j}")
                    print(
                        f"count,bdd_count: {generator.reachable[(i,j)].count()},{generator.bdd_node.count()}"
                    )
            print(count)
            solid_counts.append(count)
            # solid_counts.reverse()
        grid_of_counts.append(solid_counts)

    df = pd.DataFrame(grid_of_counts).transpose()
    heatmap = sb.heatmap(df, cmap="crest", linewidth=1, square=True, vmin=0, vmax=1)
    heatmap.set(xticklabels=[])
    heatmap.set(yticklabels=[])
    heatmap.tick_params(bottom=False, left=False)
    fig = heatmap.get_figure()
    fig.savefig("reachable_analytical_prob.png")
    fig.clf()


# TODO: analytic ERA via model counts
# generator.gen_sample_training_set("training",num_samples=10)
# height = 8
# width = 24
# generator = Platformer(height=height, width=width)

# generator.compile()
# (
#     compute_true_probs_time,
#     sample_time,
#     final_sample,
#     sample_marginal,
# ) = generator.sample()



# sat= generator.sat_sample(final_sample)
# print(f"is sat: {sat}")
# tile_size = [16, 16]
# final_image_height = height * tile_size[0]
# final_image_width = width * tile_size[1]
# final_image = Image.new("RGBA", (final_image_width, final_image_height))

# for i in range(width):
#     for j in range(height):
#         index = (i * height) + j
#         print(index)
#         print(i,j)
#         print(generator.bdd.var_at_level(index))

#         if index not in final_sample:
#             im = Image.open("non-solid.png")
#         else:
#             if final_sample[index]:
#                 im = Image.open("solid.png")
#             else:
#                 im = Image.open("non-solid.png")

            
#         final_image.paste(im, ((i * tile_size[0]), (j * tile_size[1])))


    
# final_image.save("mario.png")
# kep = ImageOps.flip(final_image)
# kep.save("post_mario.png")

# experiment = Experiment(dim=2, num_samples=1)
# metric = Metric.bit_string_metric(experiment.generator)
# experiment.add_metric_to_all_trials(metric)
# experiment.run(num_trials=1, clear_true_probs=False)
# generator = experiment.generator
# (_,_,sample,_) = generator.sample()
# assignment = generator.get_assignment(sample)
# img = generator.draw_simple_tiled(assignment)
# img.save("knots-1.png")

# height = 8
# width = 24
# generator = Platformer(height=height, width=width)
# experiment = Experiment(generator=generator, num_samples = 1000)
# metric = Metric.bit_string_metric(experiment.generator)
# experiment.add_metric_to_all_trials(metric)
# metric = Metric.hash_metric(experiment.generator)
# experiment.add_metric_to_all_trials(metric)
# experiment.run(num_trials=1, clear_true_probs=False)
# experiment.to_csv("platformer")

# generator = experiment.generator
# draw_analytical_era_platformer(generator,True)
# draw_analytical_era_reachable_platformer(generator,True)


# draw_simple_tiled_heat_map(experiment)
# draw_analytical_era_simpletiled(experiment.generator, prob=False)

# counts = generator.analytical_expressive_range_analysis()
# print(counts)
# for raw_cell,assign_cell in enumerate(generator.context.vars):

#     print(raw_cell)

# df = pd.DataFrame(counts)
# print(df)
# heatmap = sb.heatmap(df, annot=True, cmap="crest", linewidth=1)
# fig = heatmap.get_figure()
# fig.savefig("temp.png")


# draw_simple_tiled_heat_map(experiment)
# experiment.to_csv(f"experiments")
# num_iterations = 1
# num_samples = [1000,10000]
# num_samples = [1000]
# dims = [2,5,10]
# dims = [5,10]
# dims = [2]
# for i in range(num_iterations):
#     for dim in dims:
#         experiment = Experiment(dim=dim)
#         experiment.metrics = []
#         experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
#         for num_sample in num_samples:
#             experiment.run(num_samples=num_sample, num_trials = 1, recompile_per_trial=True)
#             experiment.to_csv(f"Knots-{num_sample}")


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

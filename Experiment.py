from enum import Enum
import uuid
from SimpleTiled import SimpleTiled
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from utils import get_subdirectory
from ASP import facts_from_tileset,asp_facts_from_tiles
import time
import subprocess
import random
import json


# figure out how to compute kl-divergence and jenson on samples
# var counts as metrics
# bit string
# haming distance from a given bit string


class Metric:
    def __init__(self, metric_func, metric_name):
        self.metric_name = metric_name
        self.metric_func = metric_func

    def __repr__(self):
        return self.metric_name

    def __hash__(self):
        return self.metric_name.__hash__()

    def __call__(self, *args):
        return self.metric_func(*args)

    def hamming_distance(bit_string_1, bit_string_2):
        # TODO: add proper distance
        return "this is wrong"

    def hamming_distance_metric(generator, bit_string):
        return Metric(
            lambda x: Metric.hamming_distance(x, bit_string), f"Hamming-{bit_string}"
        )

    def bit_string_metric(generator):
        return Metric(lambda x: generator.sample_as_bit_string(x), "BitString")

    def hash_metric(generator):
        return Metric(lambda x: generator.sample_as_assignment_string(x), "Hash")



class Trial:
    def __init__(
        self,
        generator,
        num_samples,
        metrics,
        recompile_per_trial,
        clear_true_probs,
    ):
        self.generator = generator
        self.num_samples = num_samples
        self.metrics = metrics
        self.recompile_per_trial = recompile_per_trial
        self.clear_true_probs = clear_true_probs
        self.metrics_data = {}
        # time_data = compile time, sample time, learning time, generating training set time
        self.trial_id = uuid.uuid4()
        self.time_data = {}

    def add_metric(self, metric):
        self.metrics.append(metric)

    def run(self):
        if self.recompile_per_trial or not self.generator.is_compiled:
            compile_time = self.generator.compile()
            print("Compiling")
            if "compile_time" not in self.time_data:
                self.time_data["compile_time"] = [compile_time]
            else:
                self.time_data["compile_time"].append(compile_time)
            print(f"Done: Compiling {compile_time / 1e09}s")

        if "sample_time" not in self.time_data:
            self.time_data["sample_time"] = []

        if "compute_true_probs_time" not in self.time_data:
            self.time_data["compute_true_probs_time"] = []

        self.metrics_data["marginal"] = []

        for i in range(0, self.num_samples):
            (
                compute_true_probs_time,
                sample_time,
                final_sample,
                sample_marginal
            ) = self.generator.sample(clear_true_probs=self.clear_true_probs)

            self.time_data["compute_true_probs_time"].append(compute_true_probs_time)
            self.time_data["sample_time"].append(sample_time)
            self.metrics_data["marginal"].append(sample_marginal)

            for metric in self.metrics:
                if metric not in self.metrics_data:
                    self.metrics_data[metric] = []
                self.metrics_data[metric].append(metric(final_sample))

    def __repr__(self):
        out_str = ""
        for el in self.__dict__:
            out_str = f"{out_str}{el}:{self.__dict__[el]}\n"

        return out_str

    def to_csv(self, path):
        # self.num_samples = num_samples
        # self.metrics = metrics
        # self.recompile_per_trial = recompile_per_trial
        # self.clear_true_probs = clear_true_probs
        # self.metrics_data = {}
        # time_data = compile time, sample time, learning time, generating training set time
        # self.time_data = {}

        print(self.metrics_data)
        filename = f"{path}/trial-{self.trial_id}.csv"

        f = open(filename, "a")

        if "sample_time" not in self.time_data:
            print("Trial did not run yet")
            return

        data_str = f"CompileTime,BDDSize,SampleTime"

        # print(self.generator.get_spec_headers())
        for header in self.generator.get_spec_headers():
            data_str += f",{header}"

        if "compute_true_probs_time" in self.time_data:
            data_str += ",ComputeProbs"


        if "marginal" in self.metrics_data:
            data_str += ",MarginalProbability"

        if "learning_time" in self.time_data:
            data_str += ",LearningTime"

        if "gen_train_time" in self.time_data:
            data_str += ",GenTrainTime"

        for metric in self.metrics:
            data_str += f",{metric}"

        data_str += "\n"

        for i in range(self.num_samples):
            compile_time = self.time_data["compile_time"][0]
            data_str += f"{compile_time}"

            bdd_size = self.generator.bdd_node.dag_size
            data_str += f",{bdd_size}"

            sample_time_data = self.time_data["sample_time"][i]
            data_str += f",{sample_time_data}"

            for header in self.generator.get_spec_headers():
                header_value = self.generator.get_header_value(header)
                data_str += f",{header_value}"

            data_str += f",{self.time_data["compute_true_probs_time"][i]}" 

            if "marginal" in self.metrics_data:
                data_str += f",{self.metrics_data["marginal"][i]}"

            if "learning_time" in self.time_data:
                learning_time_data = self.time_data["learning_time"][i]
                data_str += f",{learning_time_data}"

            if "gen_train_time" in self.time_data:
                gen_train_time_data = self.time_data["gen_train_time"][i]
                data_str += f",{gen_train_time_data}"

            for metric in self.metrics:
                data_str += f",{self.metrics_data[metric][i]}"

            data_str += "\n"

        f.write(data_str)
        f.close()

        # print(data_str)


class Experiment:
    # abstract out the experiment type/ have it be in SimpleTiled
    def __init__(self, generator=None, num_samples=1, dim=2, trials=[], metrics=[]):
        if generator == None:
            generator = SimpleTiled(dim=dim)
        self.generator = generator
        self.num_samples = num_samples
        self.trials = trials
        self.dim = dim
        self.metrics = metrics
        self.spec_heads = generator.get_spec_headers()

    def __repr__(self):
        out_str = ""
        for el in self.__dict__:
            out_str = f"{out_str}{el}:{self.__dict__[el]}\n"

        return out_str

    def run(
        self,
        num_trials=None,
        num_samples=None,
        metrics=[],
        recompile_per_trial=False,
        clear_true_probs=False,
    ):
        # clearing trials always
        self.trials.clear()

        if not num_trials:
            num_trials = len(self.trials)
            if len(self.trials) == 0:
                print("No trials ran")
                return 

        if not num_samples:
            num_samples = self.num_samples


        if len(metrics) > 0:
            self.metrics = metrics

        for i in range(num_trials):
            trial = Trial(
                self.generator,
                num_samples,
                self.metrics,
                recompile_per_trial,
                clear_true_probs,
            )
            self.trials.append(trial)

        for t in self.trials:
            t.run()

    def to_csv(self, path):
        # should return an array of the metric applied to the runs
        # we should also return the trial info somewhere in a csv
        i = 0
        test_dir = get_subdirectory(f"{path}/")
        for trial in self.trials:
            i += 1
            output_str = trial.to_csv(test_dir)

    def add_metric_to_all_trials(self, metric):
        self.metrics.append(metric)

def draw_analytical_era_simpletiled(experiment, prob=False):
    #this is a very subtle difference between this and the sample ERA
    #THE SAMPLE ERA IS A CONDITIONAL DISTRIBUTION, it forms a markov chain where the first pick you make influences all other picks.
    #This "analaytical" just counts how many solutions in total have things in that bucket
    generator = experiment.generator
    counts = []
    for i in range(generator.dim):
        rows = []
        for j in range(generator.dim):
            columns = []
            index = i * generator.dim + j
            var_name = generator.varnames[(i, j)]
            for k in range(len(generator.tile_vec)):
                new_bdd = generator.context.let({var_name: k}, generator.bdd_node)
                new_bdd_count = new_bdd.count()
                columns.append(new_bdd_count)
                if k == 12:
                    print(f"i: {i}, j:{j}, k:{k}, var_name:{var_name},new_bdd_count: {new_bdd_count}")
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
            heatmap.set_xticks(
                ticks=[i for i in range(len(counts[i][j]))],
                labels=x_labels,
                rotation=0,
            )
            heatmap.set(ylim=(1,400))

    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.savefig("wfc_analytical_era.png")


def draw_simple_tiled_samples(experiment, metric = None):
    generator = experiment.generator
    if not metric:
        metric = experiment.metrics[0]

    for trial in experiment.trials:
        for bit_string in trial.metrics_data[metric]:
            assignment = generator.assignment_from_bit_string(bit_string)
            index_assignment = generator.index_assignment_from_bit_string(bit_string)
            tile_num_assign = generator.get_assignment(index_assignment)
            final_img = generator.draw_simple_tiled(tile_num_assign)
            final_img.save(f"sample_images/{generator.spec_string()}/{bit_string}.png")


def draw_simple_tiled_heat_map(experiment, metric = None, prob=False):
    generator = experiment.generator
    if not metric:
        metric = experiment.metrics[0]

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
            print(bit_string)
            assignment = generator.assignment_from_bit_string(bit_string)
            index_assignment = generator.index_assignment_from_bit_string(bit_string)
            print(index_assignment)
            tile_num_assign = generator.get_assignment(index_assignment)
            print("title: ", tile_num_assign)

            # print(counts)
            for i, assign in enumerate(tile_num_assign):
                for j in range(len(assign)):
                    print(f"assign[j]: {assign[j]}")
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
                heatmap.set_xticks(
                    ticks=[i for i in range(len(counts[i][j]))],
                    labels=x_labels,
                    rotation=0,
                )
                heatmap.set(ylim=(1,1000))

        # fig.set_ylabel("Count")
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.savefig("wfc_era.png")



def asp_run_test(filename, output_file, ground_time, dim, sample_num):
    #this is dumb
    generator = SimpleTiled(dim=2)
    generator.compile()

    start = time.time()
    seed = int(random.random() * 10)
    cmd_str = (
        "clingo wfc.lp --seed="
        + str(seed)
        + " "
        + filename
        + "-grounded.lp -n 1 --outf=2 --sign-def=rnd --rand-freq=1"
    )
    # cmd_str = (
    #     f"clingo --seed={seed} platformer-grounded.lp -n 1 --outf=2 --sign-def=rnd --rand-freq=1"
    # )
    print(cmd_str)
    res = json.loads(subprocess.run(cmd_str, shell=True, capture_output=True).stdout)
    end = time.time()
    solve_time = res["Time"]["Total"] * 1e9
    # if solve_time == 0:
    #     solve_time = (end - start)* 1e+9
    # print("solve: ",res["Time"]["Solve"] * 1e+9)
    print((end - start) * 1e9)
    atoms = res["Call"][-1]["Witnesses"][-1]["Value"]
    for atom in atoms:
        if atom.startswith("shape("):
            shape = eval(atom[len("shape") :])


    print(atoms)
    assignment = {}
    result_tiles = np.zeros(shape)
    hash_str = ""

    beep = []
    for i in range(10):
        boop = []
        for j in range(8):
            boop.append(0)
        beep.append(boop)
            
    for atom in atoms:
        print(atom)
        if atom.startswith("assign("):
            (i, j), v = eval(atom[len("assign") :])
            result_tiles[i][j] = v

            
    for boop in beep:
        for bit in boop:
            hash_str = f"{hash_str}{bit}"

    print("assignment: ", assignment)
    # im = generator.draw_simple_tiled_asp(assignment)
    # image_name = f"asp-imgs/{filename}-{sample_num}.png"
    # im = ImageOps.mirror(im)
    # im.save(image_name)

    print(hash_str)
    output_file.write(
        "ASP," + filename + "," + str(solve_time) + "," + str(ground_time) + "," + hash_str +"\n"
    )
    return result_tiles


def draw_asp_tilemap(name="Knots", dim=2,num_samples = 1000):
    f = open("asp.csv", "a")
    gen = facts_from_tileset(name, asp_facts_from_tiles, dim)
    # print(gen)
    samples = []
    counts = []
    # for i in range(dim):
    #     rows = []
    #     for j in range(dim):
    #         columns = []
    #         for k in range(13):
    #             columns.append(0)
    #             rows.append(columns)
    #     counts.append(rows)

    for k in range(num_samples):
        start = time.time()
        res = subprocess.run(
            f"time gringo wfc.lp {name}.lp > {name}-grounded.lp",
            shell=True,
            capture_output=True,
        )
    #     # res = subprocess.run(
    #     #     f"time gringo platformer.lp > platformer-grounded.lp",
    #     #     shell=True,
        #     capture_output=True,
        # )

    #     end = time.time()
    #     ground_time = (end - start) * 1e9
    #     sample = np.array(asp_run_test(name, f, ground_time, dim,k))

    #     for i in range(dim):
    #         for j in range(dim):
    #             cell_value = sample[i][j]
    #             counts[i][j][int(cell_value)] += 1

    # count_np = np.array(counts)
    # print(count_np[0][0].sum())
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


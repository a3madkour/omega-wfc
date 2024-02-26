import sys
import json
import time
import dd.cudd as cudd
import hashlib
import uuid
import random
import os
from utils import get_subdirectory, get_all_elements_in_dir
import pickle
from enum import Enum


class SampleFormat(Enum):
    Value = 1
    Bit = 2


# state what the assumptions are for Sampling
# Also might just be a wrapper class with more functionality over omega's BDD
class SampleBDD:
    def __init__(self, bdd_node=None, filename=None):
        "docstring"
        self.weights = {}
        self.tree_true_probs = {}
        self.bdd_node = None
        self.bdd = None

        if filename:
            self.load_bdd(filename)

        if bdd_node:
            self.bdd_node = bdd_node
            self.bdd = bdd_node.bdd

        # if not self.bdd_node:
        #     sys.exit("No filename or bdd_node passed to SampleBDD")

    def gen_n_models(self,n,check_sat=False):
        models = []
        it = self.bdd.pick_iter(self.bdd_node)
        total_number_of_models =  int(self.bdd.count(self.bdd_node))
        actual_number_of_models = n
        if n > total_number_of_models:
            print(f"In gen_n_models specified n,{n},is greater than the number of models in the generator,{total_number_of_models},so we are clipping")
            actual_number_of_models = total_number_of_models


        for i in range(actual_number_of_models):
            model = next(it)
            model_bit_string = self.sample_as_bit_string(model)
            if check_sat:
                if not self.sat_omega_model(model):
                    print(f"yo we got an unsat model over here: {model}")

            models.append((model,model_bit_string))

        return (actual_number_of_models, models)



    def save_models(filename,models, append=False):
        if append:
            f = open(f"{filename}", 'a')
        else:
            f = open(f"{filename}", 'w')
        for model_tup in models:
            (model, model_bit_string) = model_tup
            f.write(f"{model_bit_string}\n")
        f.close()



    def assign_weights(self, weight=0.5):
        for var in self.bdd.vars:
            self.weights[var] = (1.0 - weight, weight)

    def compute_true_probs(self, bdd_node):
        if bdd_node == self.bdd.true:
            self.tree_true_probs[bdd_node.__hash__()] = 1.0
            return 1.0
        elif bdd_node == self.bdd.false:
            self.tree_true_probs[bdd_node.__hash__()] = 0.0
            return 0.0

        (low, high) = self.weights[bdd_node.var]
        if bdd_node.low.__hash__() not in self.tree_true_probs:
            self.compute_true_probs(bdd_node.low)

        low_tree_prob = self.tree_true_probs[bdd_node.low.__hash__()]

        if bdd_node.high.__hash__() not in self.tree_true_probs:
            self.compute_true_probs(bdd_node.high)

        high_tree_prob = self.tree_true_probs[bdd_node.high.__hash__()]

        prob_tree = (low_tree_prob * low) + (high_tree_prob * high)

        self.tree_true_probs[bdd_node.__hash__()] = prob_tree

        return prob_tree

    def sample_bdd(self, bdd_node, polarity, final_sample):
        if bdd_node == self.bdd.true:
            return 1.0
        elif bdd_node == self.bdd.false:
            return 0.0

        if bdd_node.negated:
            polarity = not polarity

        n = random.random()
        (low, high) = self.weights[bdd_node.var]

        prob_high = self.tree_true_probs[bdd_node.high.__hash__()] 
        prob_low = self.tree_true_probs[bdd_node.low.__hash__()] 


        #we need to multiply by low and high at some point

        if not polarity:
            rev_prob_high = 1.0 - prob_high
            rev_prob_low = 1.0 - prob_low
            if rev_prob_high != 0:
                #check if high is 0
                prop = (rev_prob_high) / (rev_prob_high + rev_prob_low)
            else:
                prop = 0
        else:
            prop = prob_high / (1 - 0.0 - prob_high + 1.0 - prob_low)

        if (bdd_node.high == self.bdd.true and not polarity) or (
            bdd_node.high == self.bdd.false and not polarity
        ):
            final_sample[bdd_node._index] = False
            self.sample_bdd(bdd_node.low, polarity, final_sample)
            return 1.0
        elif (
            (bdd_node.low == self.bdd.true and not polarity)
            or (bdd_node.low == self.bdd.false and not polarity)
            or (n < prop)
        ):
            final_sample[bdd_node._index] = True
            self.sample_bdd(bdd_node.high, polarity, final_sample)
            return 1.0
        else:
            final_sample[bdd_node._index] = False
            self.sample_bdd(bdd_node.low, polarity, final_sample)
            return 1.0

    def convert_binary_to_num(self, bits):
        return_value = 0
        bits.reverse()
        for bit in bits:
            return_value = (return_value << 1) + bit
        return return_value

    def sample(self, weights=None, clear_true_probs=False):
        if not weights:
            self.assign_weights()
        else:
            self.weights = weights.copy()

        if len(self.weights) != len(self.bdd_node.bdd.vars):
            sys.exit(
                f"Weights passed to sample_bdd have {len(weights)} when there are {len(self.bdd.vars)} nodes in bdd_node"
            )

        start = time.monotonic_ns()
        if clear_true_probs == True:
            self.tree_true_probs.clear()
            
        print("Compute Probs")
        self.compute_true_probs(self.bdd_node)
        end = time.monotonic_ns()
        compute_true_probs_time = end - start
        print(f"Done: compute_probs {compute_true_probs_time / 1e09}s")

        print("Sampling")
        start = time.monotonic_ns()
        final_sample = {}
        self.sample_bdd(self.bdd_node, True, final_sample)
        end = time.monotonic_ns()
        sample_time = end - start
        print(f"Done: Sampling {sample_time / 1e09}s")
        return (compute_true_probs_time, sample_time, final_sample)

    def sample_as_bit_map(self, sample):
        sample_bit_map = {}

        total_vars = len(self.bdd.vars)
        for i in range(total_vars):
            sample_bit_map[i] = int(0)

        for sample_id in sample:
            sample_bit_map[sample_id] = int(sample[sample_id])

        return sample_bit_map

    def sample_as_bit_string(self, sample):
        sample_bit_map = self.sample_as_bit_map(sample)
        sample_bit_vec = []
        for bit in sample_bit_map:
            if bit not in sample:
                sample_bit_vec.append(0)
            else:
                sample_bit_vec.append(sample[bit])

        return "".join([str(int(bit)) for bit in sample_bit_vec ])

    def dump_bdd(
        self,
        filename,
    ):
        # do the dump thing and dumb thing and output it as json and then read it bak
        # Not the best but time is a factor

        self.bdd.dump(filename, [self.bdd_node])
        out_file = open("wrapper-info-" + filename, "w")
        wrapper_info = {"weights": self.weights}
        json.dump(wrapper_info, out_file, indent=6)

    def load_bdd(self, filename):
        bdd = cudd.BDD()
        roots = bdd.load(filename)
        wrapper_file = open(f"wrapper-info-{filename}", "r")
        json_data = json.load(wrapper_file)
        self.bdd = bdd
        self.bdd_node = roots[0]
        self.weights = json_data["weights"]

    def bin_assignment(self, delta, distance):
        #TODO
        pass

    #ASSUMEs get_assignement is implemented, if I could be bothered I will add it to an interface
    def gen_sample_training_set(
        self,
        dirname,
        tile_vec,
        N,
        tile_size,
        num_samples=10,
        sample_format=SampleFormat.Bit,
    ):
        dir_path = get_subdirectory(dirname)
        assignments = []
        for i in range(num_samples):
            (_,_,sample) = self.sample()
            assignment = self.get_assignment(
                sample
            )
            assignments.append(assignment)

        # pickling for now
        # format size-num_samples-runID
        pickle_file = open(f"{dir_path}/{N}-{num_samples}-{uuid.uuid4()}.pkl", "wb")
        pickle.dump(assignments, pickle_file)
        pickle_file.close()

    def load_sample_assignments_set(self, assign_pickle_file):
        pickle_file = open(assign_pickle_file, "rb")
        assignments = pickle.load(pickle_file)
        pickle_file.close()
        return assignments

    def train_with_n_runs(self, path):
        runs = get_all_elements_in_dir(get_subdirectory(path), lambda x: "pkl" in x)
        counts = {}
        big_count = 0
        for run in runs:
            assignments = self.load_assignments_set(f"{path}/{run}")
            run_counts = self.train_with_one_run(assignments, False)
            small_count = len(assignments)
            big_count += small_count

            if len(counts) < 1:
                counts = run_counts
            else:
                for el in counts:
                    counts[el] += run_counts[el]

        for bit in counts:
            high = counts[bit] / big_count
            low = 1.0 - high
            self.weights[self.bdd.var_at_level(bit)] = (low, high)

        return counts

    def train_with_one_run(self, assignments, update_weights=True):
        counts = {}
        for i in range(len(self.weights)):
            counts[i] = 0

        for assignment in assignments:
            # print(type(assignment))
            # print(assignment)
            for i, bit in enumerate(assignment):
                # print("bit", bit)
                # print(i)
                if int(assignment[bit]) == 1:
                    counts[i] = counts[i] + 1

        if update_weights:
            for bit in counts:
                high = counts[bit] / len(assignments)
                low = 1.0 - high
                self.weights[self.bdd.var_at_level(bit)] = (low, high)

        # print(self.weights)
        # print(counts)
        return counts

    def sat_omega_model(self, model):
        cond_bdd = self.bdd.let(model, self.bdd_node)

        return self.bdd.true == cond_bdd

    def sat_sample(self, sample):
        bdd_assignment = {self.bdd.var_at_level(k): v for k, v in sample.items()}

        return self.sat_omega_model(bdd_assignment)


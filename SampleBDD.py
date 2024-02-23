import sys
import json
import dd.cudd as cudd
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

        if not self.bdd_node:
            sys.exit("No filename or bdd_node passed to SampleBDD")

    def assign_weights(self, weight=0.5):
        for var in self.bdd.vars:
            self.weights[var] = (1.0 - weight, weight)

    def compute_true_probs(self, bdd_node, clear=False):
        # TODO: okay we need to figure out how to deal with overflow issues
        if clear:
            self.tree_true_probs.clear()

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
        n = random.random()
        (low, high) = self.weights[bdd_node.var]
        prob_high = self.tree_true_probs[bdd_node.high.__hash__()] * high
        prob_low = self.tree_true_probs[bdd_node.low.__hash__()] * low
        polarity = polarity
        if bdd_node.negated:
            polarity = not polarity

        if not polarity:
            rev_prob_high = 1.0 - prob_high
            rev_prob_low = min(1.0 - prob_low, 0.9)
            prop = (rev_prob_high) / (rev_prob_high + rev_prob_low)
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

    def sample(self, weights=None):
        if not weights:
            self.assign_weights()
        else:
            self.weights = weights.copy()

        if len(self.weights) != len(self.bdd_node.bdd.vars):
            sys.exit(
                f"Weights passed to sample_bdd have {len(weights)} when there are {len(self.bdd.vars)} nodes in bdd_node"
            )

        self.compute_true_probs(self.bdd_node)
        final_sample = {}
        self.sample_bdd(self.bdd_node, True, final_sample)
        return final_sample

    def sample_as_bit_string(self, sample):
        sample_bit_string = [0] * len(self.bdd.support(self.bdd_node))
        for assign in sample:
            # print(assign)
            sample_bit_string[assign] = int(sample[assign])

        return sample_bit_string

    def get_assignment(
        self, sample, tile_vec, dim, tile_size, sample_format=SampleFormat.Value
    ):
        num_bits = (self.bdd._number_of_cudd_vars() / dim) / dim
        final_assignment = []
        sample_bit_string = self.sample_as_bit_string(sample)

        if sample_format == SampleFormat.Bit:
            return sample_bit_string

        for i in range(0, dim):
            final_assignment.append([])
            for j in range(0, dim):
                current_index = int(i * dim * num_bits + j * num_bits)
                end_index = int(current_index + num_bits)
                # print(
                #     self.convert_binary_to_num(
                #     sample_bit_string[current_index:end_index]
                # )
                # )
                final_assignment[i].append(
                    self.convert_binary_to_num(
                        sample_bit_string[current_index:end_index]
                    )
                )

        return final_assignment

    # def draw_sample(self,sample):
    #     num_bits = (self.bdd._number_of_cudd_vars() / N) / N
    # final_assignment = []
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

    def hash_assignment(self, assignment):
        pass

    def gen_uniform_training_set(
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
            sample = self.sample()
            assignment = self.get_assignment(
                sample, tile_vec, N, tile_size, sample_format
            )
            assignments.append(assignment)

        # pickling for now
        # format size-num_samples-runID
        pickle_file = open(f"{dir_path}/{N}-{num_samples}-{uuid.uuid4()}.pkl", "wb")
        pickle.dump(assignments, pickle_file)
        pickle_file.close()

    def load_assignments_set(self, assign_pickle_file):
        pickle_file = open(assign_pickle_file, "rb")
        assignments = pickle.load(pickle_file)
        pickle_file.close()
        return assignments

    def train_with_n_runs(self, path):
        runs = get_all_elements_in_dir(get_subdirectory(path), lambda x : "pkl" in x)
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
            for i, bit in enumerate(assignment):
                if bit == 1:
                    counts[i] = counts[i] + 1

        if update_weights:
            for bit in counts:
                high = counts[bit] / len(assignments)
                low = 1.0 - high
                self.weights[self.bdd.var_at_level(bit)] = (low, high)

        return counts

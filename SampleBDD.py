import sys
import random


# state what the assumptions are for Sampling
# Also might just be a wrapper class with more functionality over omega's BDD
class SampleBDD:
    def __init__(self, bdd_node):
        "docstring"
        self.bdd_node = bdd_node
        self.bdd = bdd_node.bdd
        self.tree_true_probs = {}

    def assign_weights(self, weight=0.5):
        weights = {}
        for var in self.bdd.vars:
            weights[var] = (1.0 - weight, weight)
        return weights

    def compute_true_probs(self, bdd_node, weights):
        # TODO: okay we need to figure out how to deal with overflow issues
        if bdd_node == self.bdd.true:
            self.tree_true_probs[bdd_node] = 1.0
            return 1.0
        elif bdd_node == self.bdd.false:
            self.tree_true_probs[bdd_node] = 0.0
            return 0.0

        (low, high) = weights[bdd_node.var]
        if bdd_node.low not in self.tree_true_probs:
            self.compute_true_probs(bdd_node.low, weights)

        low_tree_prob = self.tree_true_probs[bdd_node.low]

        if bdd_node.high not in self.tree_true_probs:
            self.compute_true_probs(bdd_node.high, weights)

        high_tree_prob = self.tree_true_probs[bdd_node.high]

        prob_tree = (low_tree_prob * low) + (high_tree_prob * high)

        self.tree_true_probs[bdd_node] = prob_tree

        return prob_tree

    def sample_bdd(self, bdd_node,polarity, weights, final_sample):
        if bdd_node == self.bdd.true:
            return 1.0
        elif bdd_node == self.bdd.false:
            return 0.0
        n = random.random()
        (low, high) = weights[bdd_node.var]
        prob_high = self.tree_true_probs[bdd_node.high] * high
        prob_low = self.tree_true_probs[bdd_node.low] * low
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
            self.sample_bdd(
                bdd_node.low, polarity, weights, final_sample
            )
            return 1.0
        elif (
            (bdd_node.low == self.bdd.true and not polarity)
            or (bdd_node.low == self.bdd.false and not polarity)
            or (n < prop)
        ):
            final_sample[bdd_node._index] = True
            self.sample_bdd(
                bdd_node.high, polarity, weights, final_sample
            )
            return 1.0
        else:
            final_sample[bdd_node._index] = False
            self.sample_bdd(
                bdd_node.low,  polarity, weights, final_sample
            )
            return 1.0

    def convert_binary_to_num(self, bits):
        return_value = 0
        bits.reverse()
        for bit in bits:
            return_value = (return_value << 1) + bit
        return return_value

    def sample(self, weights=None):
        if not weights:
            weights = self.assign_weights()
        if len(weights) != len(self.bdd_node.bdd.vars):
            sys.exit(
                f"Weights passed to sample_bdd have {len(weights)} when there are {len(self.bdd.vars)} nodes in bdd_node"
            )

        self.compute_true_probs(self.bdd_node, weights)
        final_sample = {}
        self.sample_bdd(self.bdd_node, True, weights, final_sample)
        return final_sample

    def sample_as_bit_string(self, sample):
        sample_bit_string = [0] * len(self.bdd.support(self.bdd_node))
        for assign in sample:
            # print(assign)
            sample_bit_string[assign] = int(sample[assign])

        return sample_bit_string

    def get_assignment(self, sample, tile_vec, dim, tile_size):
        num_bits = (self.bdd._number_of_cudd_vars() / dim) / dim
        final_assignment = []
        sample_bit_string = self.sample_as_bit_string(sample)
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

    def dump_bdd(self, filename, weights=None, true_tree_probs=None, dump_weights=False, dump_tree_probs=False):
        pass

    def load_bdd(self, filename):
        pass

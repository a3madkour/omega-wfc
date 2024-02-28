from SampleBDD import SampleBDD, SampleFormat
from omega.symbolic.fol import Context
from collections import defaultdict
from functools import cache
import time

class Platformer(SampleBDD):
    def __init__(self, width = 24, height=8):
        super().__init__()
        self.width = width
        self.name = "Platformer"
        self.height = height
        self.is_compiled = False
        self.tile_vec = None
        self.tile_size = None
        self.reachable = None
        self.var_names = None

    
    def compile(self):

        start = time.monotonic_ns()
        var_names = {(i,j): f"solid_{i}_{j}" for i in range(self.width) for j in range(self.height)}
        self.var_names = var_names

        context = Context()
        context.declare(
            **{v: 'bool' for v in var_names.values()}
        )

        context.bdd.configure(reordering=False)

        solid = defaultdict(lambda: context.false)
        solid.update({k: context.add_expr(v) for k,v in var_names.items()})

        reachable = defaultdict(lambda: context.false)

        for j in range(self.height-1):
            reachable[(0,j)] = context.false

        reachable[(0,self.height-1)] = context.true
        for i in range(1,self.width):
            for j in range(self.height):
                reachable[(i,j)] = context.false # not reachable by default

                floor_below = solid.get((i,j+1), context.true)

                # horizontal walk
                reachable[(i,j)] |= reachable[(i-1,j)] & floor_below & ~solid[(i,j)]

                # horizontal jump across gap
                reachable[(i,j)] |= reachable[(i-2,j)] & ~solid[(i-1,j)] & ~solid[(i-1,j-1)] & floor_below & ~solid[(i,j)]

                # step up
                reachable[(i,j)] |= reachable[(i-1,j+1)] & floor_below & ~solid[(i,j)] & ~solid[(i-1,j)]

                # fall down (extended knight's move downward)
                for k in range(j):
                    free = context.true
                    for kk in range(k,j+1):
                        free &= ~solid[(i,kk)]
                    reachable[(i,j)] |= reachable[(i-1,k)] & free & floor_below

        playable = reachable[(self.width-1,0)]

        spec = playable & ~solid[(0,self.height-1)]

        self.bdd_node = spec
        self.bdd = spec.bdd
        self.context = context
        self.reachable = reachable
        
        end = time.monotonic_ns()
        end_time = end - start

        self.is_compiled = True
        return end_time

    def get_assignment(
        self, sample, sample_format=SampleFormat.Value
    ):

        num_bits = (self.bdd._number_of_cudd_vars() / self.width) / self.width
        final_assignment = []
        sample_bit_vec = []

        for bit in self.sample_as_bit_map(sample):
            if bit not in sample:
                sample_bit_vec.append(0)
            else:
                sample_bit_vec.append(sample[bit])

        if sample_format == SampleFormat.Bit:
            return sample_bit_vec

        print(sample_bit_vec)
        for i in range(0, self.dim):
            final_assignment.append([])
            for j in range(0, self.dim):
                current_index = int(i * self.dim * num_bits + j * num_bits)
                end_index = int(current_index + num_bits)
                # print(
                #     self.convert_binary_to_num(
                #     sample_bit_vec[current_index:end_index]
                # )
                # )
                final_assignment[i].append(
                    self.convert_binary_to_num(sample_bit_vec[current_index:end_index])
                )

        return final_assignment

    def get_spec_headers(self):
        return ["Width", "Height"]

    def get_header_value(self, header):
        #tieing too much data in the code here but we are short on time
        if header == "Width":
            return self.width
        elif header == "Height":
            return self.height



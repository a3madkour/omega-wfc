from SampleBDD import SampleBDD, SampleFormat
from omega.symbolic.fol import Context
from collections import defaultdict
import seaborn as sb
import pandas as pd
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

        #a dict with default value context.false
        solid = defaultdict(lambda: context.false)
        solid.update({k: context.add_expr(v) for k,v in var_names.items()})

        reachable = defaultdict(lambda: context.false)

        for j in range(self.height-1):
            reachable[(0,j)] = context.false

        #the tile atop the beginning tile is reachable
        reachable[(0,self.height-1)] = context.true
        for i in range(1,self.width):
            for j in range(self.height):
                reachable[(i,j)] = context.false # not reachable by default

                #is the floor below me true?
                floor_below = solid.get((i,j+1), context.true)

                # horizontal walk-- I can get to i,j from i-1,j
                reachable[(i,j)] |= reachable[(i-1,j)] & floor_below & ~solid[(i,j)]

                # horizontal jump across gap -- I can get to i,j from i-2,j
                reachable[(i,j)] |= reachable[(i-2,j)] & ~solid[(i-1,j)] & ~solid[(i-1,j-1)] & floor_below & ~solid[(i,j)]

                # step up -- I can jump to i,j from i-1,j+1 (i.e diagonal right (+x(from i-1)) and up (-y(from j+1)) )
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

        final_assignment = {}
        for index in sample:
            final_assignment[self.bdd.var_at_level(index)] = sample[index]

        return final_assignment

    def get_spec_headers(self):
        return ["Width", "Height"]

    def get_header_value(self, header):
        #tieing too much data in the code here but we are short on time
        if header == "Width":
            return self.width
        elif header == "Height":
            return self.height

    def analytical_expressive_range_analysis(self, expr=None):
        # we need to iterate over all the possbilites right?
        # So how many times does i,j cell be tile t
        # TODO: Figure out how to use the cached counts tree_true_probs

        era_counts = {}
        # each location in the grid
        assign_count = {}

        bdd_node = self.bdd_node
        if expr:
            bdd_node = self.bdd_node & expr

        for assign_cell in self.context.vars:
            kop = self.context.vars[assign_cell]
            assign_cell_count = []
            assignment = {str(assign_cell): True}
            rs = self.context.let(assignment, bdd_node)
            co = int(rs.count())
            assign_cell_count.append(co)
            assign_count[assign_cell] = assign_cell_count

            # print(assign_cell)
        return assign_count


def draw_analytical_era_reachable_platformer(generator, prob=False):
    # countable models where a given tile is reachable
    grid_of_counts = []
    for i in range(generator.width):
        solid_counts = []
        for j in range(generator.height):
            reach_bdd_spec= generator.reachable[(i, j)]
            actual_node_to_count = reach_bdd_spec & generator.bdd_node
            count = actual_node_to_count.count()

            if prob:
                count = count / generator.bdd_node.count()
                if count > 1:
                    print("how")
                    print(f"i,j: {i},{j}")
                    print(
                        f"count,bdd_count: {generator.reachable[(i,j)].count()},{generator.bdd_node.count()}"
                    )
            print(f"(i,j): ({i},{j}), count:{count}")
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


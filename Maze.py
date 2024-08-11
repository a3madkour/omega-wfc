from SampleBDD import SampleBDD, SampleFormat
import uuid
from utils import get_subdirectory, get_all_elements_in_dir,props_from_graph, maze_props, sol_props 
import random
from omega.symbolic.fol import Context
from collections import defaultdict
import time
import pickle
import networkx as nx
import matplotlib.pyplot as plt


class Maze(SampleBDD):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def compile(self):
        start = time.monotonic_ns()
        var_names = {}
        var_values = {}
        maze_size = self.dim
        for i in range(maze_size):
            for j in range(maze_size):
                current_idx = (i * maze_size) + j
                if j + 1 < maze_size:
                    var_names[(current_idx, current_idx + 1)] = (
                        f"edge{current_idx}_{current_idx+1}"
                    )
                    var_values[f"edge{current_idx}_{current_idx+1}"] = (
                        current_idx,
                        current_idx + 1,
                    )
                if current_idx + maze_size < maze_size * maze_size:
                    var_names[(current_idx, current_idx + maze_size)] = (
                        f"edge{current_idx}_{current_idx+maze_size}"
                    )
                    var_values[f"edge{current_idx}_{current_idx+maze_size}"] = (
                        current_idx,
                        current_idx + maze_size,
                    )

        self.var_values = var_values
        self.var_names = var_names
        context = Context()
        context.declare(**{v: "bool" for v in var_names.values()})

        context.bdd.configure(reordering=False)
        edge_exist = defaultdict(lambda: context.false)
        edge_exist.update({k: context.add_expr(v) for k, v in var_names.items()})
        reachable = defaultdict(lambda: context.false)

        for i in range(maze_size**2):
            reachable[i] = context.false
        # first node is reachable
        reachable[0] = context.true

        # we could probably make this faster by using let--similar to WFC
        for i in range(maze_size):
            for j in range(maze_size):
                current_idx = (i * maze_size) + j
                if current_idx - 1 >= 0:
                    reachable[current_idx] |= (
                        reachable[current_idx - 1]
                        & edge_exist[(current_idx - 1, current_idx)]
                    )
                if current_idx - maze_size >= 0:
                    reachable[current_idx] |= (
                        reachable[current_idx - maze_size]
                        & edge_exist[(current_idx - maze_size, current_idx)]
                    )

        spec = reachable[(maze_size**2) - 1]
        self.bdd_node = spec
        self.bdd = spec.bdd
        self.context = context
        self.reachable = reachable
        end = time.monotonic_ns()
        end_time = end - start
        self.is_compiled = True
        # print(spec.count())
        return end_time

    def get_assignment(self, sample, sample_format=SampleFormat.Value):

        final_assignment = {}
        for index in range(len(self.bdd.vars)):
            if index in sample:
                final_assignment[self.bdd.var_at_level(index)] = sample[index]
            else:
                rnd = random.random()
                value = False
                # if rnd > 0.5:
                #     value = True
                final_assignment[self.bdd.var_at_level(index)] = value

        return final_assignment

    def get_maze_graph(self, assignment):
        edge_list = []
        G = nx.Graph()
        for assign in assignment:
            if assignment[assign]: 
                edge = self.var_values[assign]
                u,v = edge
                G.add_edge(u,v,weight =1)
        return G

    def draw_maze(self, maze_graph, filename="maze.png"):
        pos = {((i*self.dim)+j): [j,-i] for i in range(self.dim) for j in range (self.dim)}
        # nx.draw(maze_graph,pos, labels = node_type)
        nx.draw(maze_graph,pos)
        
        # nx.draw(maze_graph)
        plt.savefig(filename)

    def gen_sample_training_set(
        self,
        dirname,
        num_samples=10,
        sample_format=SampleFormat.Bit,
    ):
        dir_path = get_subdirectory(dirname)
        assignments = []
        for i in range(num_samples):
            (_, _, sample, _) = self.sample()
            assignment = self.get_assignment(sample)
            graph = self.get_maze_graph(assignment)
            cell_type = props_from_graph(graph,self.dim)
            maze_prop = maze_props(cell_type)
            #TODO: hard-coding condition this for now
            sol_prop = sol_props(graph,self.dim, cell_type)
            # print(sol_prop)
            # assignments.append(assignment)
            # selected_edge = self.var_names[(0,1)]  
            # if assignment[selected_edge]:
            #     assignments.append(assignment)
            # print(selected_edge)
            # print(sol_prop)
            if 'tu' in sol_prop:
                if sol_prop['tu'] >= 5:
                    assignments.append(assignment)

        # pickling for now
        # format size-num_samples-runID
        file_name = f"{num_samples}-{uuid.uuid4()}.pkl"
        pickle_file = open(f"{dir_path}/{file_name}", "wb")
        pickle.dump(assignments, pickle_file)
        pickle_file.close()
        return file_name

    def load_sample_assignments_set(self, assign_pickle_file):
        pickle_file = open(assign_pickle_file, "rb")
        assignments = pickle.load(pickle_file)
        pickle_file.close()
        return assignments


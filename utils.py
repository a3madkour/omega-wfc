from os import mkdir, path, getcwd,listdir
from collections import Counter
import networkx as nx

def get_subdirectory(sd):
    dir = path.join(getcwd(), sd)
    if not path.isdir(dir):
        mkdir(dir)
    return dir

def get_all_elements_in_dir(sd, filter=lambda x: x):
    return  [file for file in listdir(sd) if filter(file)]

def props_from_graph(graph: nx.Graph, maze_size):
    cell_type = {}
    for node in graph.nodes:
        edges = graph.edges(node)
        n_non_zero_neighs = 0
        top_bot = False
        right_left = False
        for edge in edges:
            u,v = edge
            data = graph.get_edge_data(u,v)
            weight = data['weight']
            if weight > 1:
                pass
                # print(edge)
                # print("Infinite")
            else:
                n_non_zero_neighs = n_non_zero_neighs + 1
                if int(v) - int(u) == 1 or int(u) - int(v) == 1 :
                    right_left = True
                    # print("rightleft")
                elif int(v) - int(u) == maze_size or int(u) - int(v) == maze_size:
                    top_bot = True
                    # print("top_bot")
                # print(edge)
                # print("not infinite")
        if n_non_zero_neighs == 1:
            cell_type[node] = "tr"
        elif n_non_zero_neighs == 2:
            if top_bot:
                if right_left:
                    cell_type[node] = "tu"
                else:
                    cell_type[node] = "s"
            elif right_left:
                cell_type[node] = "s"
            else:
                print("this can't happen right?")
        elif n_non_zero_neighs == 3:
            cell_type[node] = "T"
        elif n_non_zero_neighs == 4:
            cell_type[node] = "c"
    return cell_type
 
def maze_props(cell_type):
    maze_low_level_props = Counter(cell_type.values())
    return maze_low_level_props

def sol_props(G,maze_size,cell_type):
    st = 0
    nd = (maze_size * maze_size) - 1  
    sol_path = nx.shortest_path(G, st,nd, weight="weight")
    sol_path_props = {"tu":0, "s":0, "d":0, "len":0}
    for nod in sol_path:
        if nod not in cell_type:
            continue

        if cell_type[nod] == "tu":
            sol_path_props["tu"] = sol_path_props["tu"] + 1
        if cell_type[nod] == "s":
            sol_path_props["s"] = sol_path_props["s"] + 1
        if cell_type[nod] == "T" or cell_type[nod] == "c":
            sol_path_props["d"] = sol_path_props["d"] + 1
    sol_path_props["len"] = len(sol_path)
    return sol_path_props


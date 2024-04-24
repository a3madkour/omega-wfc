import json
import numpy as np

def facts_from_tileset(path, tiles2facts, dim):

    """Generate facts from tileset json file at *path* using the *tiles2facts* function and writes them to a file 

    :param path: Path of the JSON file for the tileset
    :param tiles2facts: Function used to convert tileset to facts
    :param dim: Dimension of the grid
    :returns: 

    """
    file = open(f"tileset-json/{path}-patterns.json")
    data = json.load(file)
    tile_vec = data["tile_vec"]
    h_adj = data["patterns"]["x_axis"]
    v_adj = data["patterns"]["y_axis"]

    
    with open( path+ '.lp', 'w') as f:
        for fact in tiles2facts(h_adj,v_adj,tile_vec,dim):
            f.write(fact + '\n')


def asp_facts_from_tiles(h_adj,v_adj,tile_vec,dim):

    """Convert tile adjacency patterns into ASP facts

    :param h_adj: List of horizontal patterns
    :param v_adj: List of vertical patterns
    :param tile_vec: Dictionary of tile information
    :param dim: Dimension of the grid
    :returns: List of ASP facts from the given patterns

    """
    facts = []

    for entry in tile_vec:
        facts.append(f'tile({entry["index"]}).')

    
    shape = (dim, dim)
    cells = list(np.ndindex(shape))

    facts.append('shape('+repr(shape).replace(' ','')+').')

    for cell in cells:
        facts.append('cell('+repr(cell).replace(' ','')+').')

    for i in range(dim ):
        for j in range(dim):
            if i < dim - 1:
                facts.append(f"adj(0,({i},{j}), ({i+1},{j})).")
            if j < dim -1:
                facts.append(f"adj(1,({i},{j}), ({i},{j+1})).")

    for adj in h_adj:
        # print(adj)
        c1 = adj[0]
        c2 = adj[1]
        facts.append(f'pair_ppt({0},'+\
                            repr(c1).replace(' ','')+','+\
                            repr(c2).replace(' ','')+',1000).')
    for adj in v_adj:
        c1 = adj[0]
        c2 = adj[1]


        facts.append(f'pair_ppt({1},'+\
                            repr(c1).replace(' ','')+','+\
                            repr(c2).replace(' ','')+',1000).')


    return facts


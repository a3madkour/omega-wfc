import json
import numpy as np
from collections import  Counter

def facts_from_tileset(path, tiles2facts, dim):

    file = open(f"tileset-json/{path}-patterns.json")
    data = json.load(file)
    tile_vec = data["tile_vec"]
    h_adj = data["patterns"]["x_axis"]
    v_adj = data["patterns"]["y_axis"]

    

    # filepath = "mxgmn-wfc/samples/"+filename+".png"
    # img = imread(filepath)#.transpose((1,0,2))
    # dim= img.shape[2]
    # tile_hashes = img @ (256**np.arange(dim))
    # unique_hashes = np.unique(tile_hashes)
    # tiles = np.searchsorted(unique_hashes, tile_hashes)
    # print(tiles)
    with open( path+ '.lp', 'w') as f:
        for fact in tiles2facts(h_adj,v_adj,tile_vec,dim):
            f.write(fact + '\n')

    # return unique_hashes,tiles

def asp_facts_from_tiles(h_adj,v_adj,tile_vec,dim):

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

    print(facts)

    return facts


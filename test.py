from omega.symbolic.fol import Context
import json
import time

name = "Knots"
file = open(f'{name}_pattern.json')

data = json.load(file)

N = 5

h_adj = data["x_axis"]
v_adj = data["y_axis"]

T = data["num_colors"]

variables = {}
locations = {}
for i in range(N):
  for j in range(N):
    var = f"assign_{i}_{j}"
    variables[var] = (0,T-1)
    locations[var] = (i,j)

context = Context()
context.declare(**variables)
context.bdd.configure(reordering=False);




one_cell = context.true
h_cases = context.false
start = time.monotonic_ns()
for u,v in h_adj:
    h_cases |= context.add_expr(f"assign_0_0 = {u} & assign_1_0 = {v}")
    v_cases = context.false
    for u,v in h_adj:
        v_cases |= context.add_expr(f"assign_0_0 = {u} & assign_0_1 = {v}")

h_cases_shifted = context.let({'assign_0_0': 'assign_0_1', 'assign_1_0': 'assign_1_1'}, h_cases)
v_cases_shifted = context.let({'assign_0_0': 'assign_1_0', 'assign_0_1': 'assign_1_1'}, v_cases)

one_cell &= h_cases
one_cell &= v_cases
one_cell &= h_cases_shifted
one_cell &= v_cases_shifted

generator = context.true
dag_sizes = []
dag_sizes.append(generator.dag_size)

for i in range(N-1):
    for j in range(N-1):
        if i == 0 and j == 0:
            generator &= one_cell
        else:
            #the let is as follows; replace the thing before the : with the thing after in the operator that is the second argument, in this case one_cell. So we are saying rename assign_0_0 to assign_i_j in the formula one_cell and return the result. We then and this to the previously iterated on generator to get a generator with an extra cell
            generator &= context.let({'assign_0_0': f'assign_{i}_{j}',
                                  'assign_0_1': f'assign_{i}_{j+1}',
                                  'assign_1_0': f'assign_{i+1}_{j}',
                                  'assign_1_1': f'assign_{i+1}_{j+1}'
                                  }, one_cell)
            dag_sizes.append(generator.dag_size)


end = time.monotonic_ns()
print("time:", end-start)
#print(generator.dag_size)

gen_bdd = generator.bdd

bdd_file = f'bdds/{name}_{N}x{N}_bdd.json'
gen_bdd.dump(bdd_file, [generator])

from omega.symbolic.fol import Context
import json
import time
import random
import uuid
from os import mkdir, path, getcwd
# from draw import draw_sample
    

from SampleBDD import SampleBDD
from SimpleTiled import draw_simple_tiled
       

def get_subdirectory(sd):
    dir = path.join(getcwd(), sd)
    if not path.isdir(dir):
        mkdir(dir)
    return dir


# names =  ["Castle", "Circles","Circuit", "FloorPlan", "Knots", "Rooms", "Summer"]
names = ["Knots"]
# sizes = [2,5]
sizes = [2]
for name in names:
    for N in sizes:
        # print("We are at: (name:", name, ",N:",N,")")
        file = open(f"tileset-json/{name}-patterns.json")
        csv_file = open(f"csvs/{name}_compile_times.csv", "a")
        data = json.load(file)
        tile_vec = data["tile_vec"]

        h_adj = data["patterns"]["x_axis"]
        v_adj = data["patterns"]["y_axis"]

        tile_size = data["patterns"]["tile_size"]

        # print("h_adj",len(h_adj))
        # print("v_adj",len(v_adj))

        T = data["patterns"]["num_colors"]
        # print(T)

        variables = {}
        locations = {}
        for i in range(N):
            for j in range(N):
                var = f"assign_{i}_{j}"
                variables[var] = (0, T - 1)
                locations[var] = (i, j)
        context = Context()
        context.declare(**variables)
        context.bdd.configure(reordering=False)
        # print(variables)

        start = time.monotonic_ns()
        one_cell = context.true
        h_cases = context.false
        for u, v in h_adj:
            h_cases |= context.add_expr(f"assign_0_0 = {u} & assign_1_0 = {v}")

        v_cases = context.false
        for u, v in v_adj:
            v_cases |= context.add_expr(f"assign_0_0 = {u} & assign_0_1 = {v}")

        h_cases_shifted = context.let(
            {"assign_0_0": "assign_0_1", "assign_1_0": "assign_1_1"}, h_cases
        )
        v_cases_shifted = context.let(
            {"assign_0_0": "assign_1_0", "assign_0_1": "assign_1_1"}, v_cases
        )

        one_cell &= h_cases
        one_cell &= v_cases
        one_cell &= h_cases_shifted
        one_cell &= v_cases_shifted

        num_sols = context.count(one_cell)
        example_sol = context.pick(one_cell)
        # print("example_sol:", example_sol)
        # print("num_sols:", num_sols)
        generator = context.true
        dag_sizes = []
        dag_sizes.append(generator.dag_size)

        for i in range(N - 1):
            for j in range(N - 1):
                if i == 0 and j == 0:
                    generator &= one_cell
                else:
                    # the let is as follows; replace the thing before the : with the thing after in the operator that is the second argument, in this case one_cell. So we are saying rename assign_0_0 to assign_i_j in the formula one_cell and return the result. We then and this to the previously iterated on generator to get a generator with an extra cell
                    generator &= context.let(
                        {
                            "assign_0_0": f"assign_{i}_{j}",
                            "assign_0_1": f"assign_{i}_{j+1}",
                            "assign_1_0": f"assign_{i+1}_{j}",
                            "assign_1_1": f"assign_{i+1}_{j+1}",
                        },
                        one_cell,
                    )
                    dag_sizes.append(generator.dag_size)

        end = time.monotonic_ns()
        end_time = end - start
        # print(generator.dag_size)

        gen_bdd = generator.bdd
        # total_sol_count = context.count(generator)
        # assignment_file = open("samples_assignment.txt", "w")
        # if total_sol_count < 3000:
        #     sample_iter = context.pick_iter(generator)
        #     for i in range(100):
        #         one_sample = next(sample_iter)
        #         final_image = draw_sample(one_sample, tile_vec, N, tile_size)
        #         # final_image_path = f"{get_subdirectory(f'samples/{name}')}/{N}x{N}--{uuid.uuid4()}.png"
        #         final_image_path = (
        #             f"{get_subdirectory(f'samples/{name}')}/{N}x{N}--{i}.png"
        #         )
        #         assignment_file.write(str(i) + ":" + str(one_sample) + "\n")
        #         final_image.save(final_image_path)

        # print(gen_bdd)

        # sample_iter = context.pick_iter(generator)
        # print(sample_iter)
        # print(next(sample_iter))
        # print(next(sample_iter))

        # final_image = draw_sample(one_sample, tile_vec, N, tile_size)
        # final_image_path = f"{get_subdirectory(f'samples/{name}')}/{N}x{N}--{uuid.uuid4()}.png"
        # final_image.save(final_image_path)
        # bdd_file = f'bdds/{name}_{N}x{N}_bdd.json'
        # csv_file.write(f'{N},{end_time},{generator.dag_size}\n')
        # csv_file.close()
        # print("dumping...")
        # gen_bdd.dump(bdd_file, [generator])
        # print("done dumping")
        sol = context.pick(generator)
        # print("sample sol:", sol)
        # for v in context.vars:
        #     print(context.vars[v])

        # print(gen_bdd.succ(generator))

        # var_level is an index in the vars?array for each node in the bdd
        # var_level is an index in the vars?array for each node in the bdd

        # TODO we need to come up with a bottom up mapping of the BDD so we can "uniformly sample" from true

        # start with laying out the bdd. I.e. make a dict/struct that will store the parents of a node. So when we query for parents we get a list of node refs

        external_dict = {}
        root_props = gen_bdd.succ(generator)
        # print(generator._index)

        print("done compiling")

        bdd_wrapper = SampleBDD(generator)
        sample = bdd_wrapper.sample()
        assignment = bdd_wrapper.get_assignment(sample,tile_vec,N,tile_size)
        final_img = draw_simple_tiled(assignment,tile_vec,N,tile_size)

        bdd_wrapper.dump_bdd("temp.json")

        new_wrapper = SampleBDD(filename="temp.json")



        # print(dir(generator))
        # print(dir(gen_bdd))
        # reverse_bdd_node = ReverseBDDNode(
        #     root_props[0], gen_bdd.var_at_level(root_props[0]), generator, {}, gen_bdd
        # )

        # reverse_bdd_node.traverse(external_dict)
        # reverse_node_true = external_dict[gen_bdd.true]
        # assignment = {}
        # uniform_sample(reverse_node_true, external_dict, assignment)

        # sample_bit_string = [0] * len(gen_bdd.support(generator))
        # for assign in assignment:
        #     print(assign)
        #     sample_bit_string[assign] = assignment[assign]

        # # print(sample_bit_string)
        # print(assignment)
        # num_bits = (len(gen_bdd.support(generator)) / N) / N

        # final_assignment = []
        # print(final_assignment)
        # print(sample_bit_string)
        # for i in range(0, N):
        #     final_assignment.append([])
        #     for j in range(0, N):
        #         current_index = int(i * N * num_bits + j * num_bits)
        #         end_index = int(current_index + num_bits)
        #         print(convert_binary_to_num(sample_bit_string[current_index:end_index]))
        #         final_assignment[i].append(
        #             convert_binary_to_num(sample_bit_string[current_index:end_index])
        #         )

        # final_img = draw_sample(final_assignment, tile_vec, N, tile_size)


        # sample_bdd(generator)
        # weights = assign_weights(generator)
        # print(weights)

        # tree_true_probs = {}
        # compute_true_probs(generator, weights, tree_true_probs)

        # final_sample = {}
        # sample_bdd(generator, tree_true_probs,True,final_sample)


        # print(final_sample)

        # sample_bit_string = [0] * len(gen_bdd.support(generator))
        # for assign in final_sample:
            # print(assign)
            # sample_bit_string[assign] = int(final_sample[assign])

        # print(sample_bit_string)
        # print(assignment)
        # print(dir(gen_bdd))
        # num_bits = (gen_bdd._number_of_cudd_vars() / N) / N

        # final_assignment = []
        # print(final_assignment)
        # print(sample_bit_string)
        # for i in range(0, N):
        #     final_assignment.append([])
        #     for j in range(0, N):
        #         current_index = int(i * N * num_bits + j * num_bits)
        #         end_index = int(current_index + num_bits)
        #         print(convert_binary_to_num(sample_bit_string[current_index:end_index]))
        #         final_assignment[i].append(
        #             convert_binary_to_num(sample_bit_string[current_index:end_index])
        #         )

        # print(final_assignment)

        # final_img = draw_sample(final_assignment, tile_vec, N, tile_size)



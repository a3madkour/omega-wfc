from PIL import Image, ImageOps
from omega.symbolic.fol import Context
import time
import numpy as np
from enum import Enum
import sys
import json
from SampleBDD import SampleBDD, SampleFormat

# tile_counts


class TileSet(Enum):
    Knots = "Knots"
    Castle = "Castle"
    Circles = "Circles"
    FloorPlan = "FloorPlan"
    Rooms = "Rooms"
    Circuit = "Circuit"


class SimpleTiled(SampleBDD):
    def __init__(self, tileset=TileSet.Knots, dim=2, path="tileset-json"):
        super().__init__()
        self.tileset = tileset
        self.dim = dim
        self.path = path
        self.name = tileset.name
        self.is_compiled = False
        self.tile_vec = None
        self.tile_size = None
        self.varnames = {}

    def compile(self):
        file = open(f"{self.path}/{self.name}-patterns.json")
        data = json.load(file)
        self.tile_vec = data["tile_vec"]
        h_adj = data["patterns"]["x_axis"]
        v_adj = data["patterns"]["y_axis"]

        self.tile_size = data["patterns"]["tile_size"]

        # print("h_adj",len(h_adj))
        # print("v_adj",len(v_adj))

        T = data["patterns"]["num_colors"]
        # print(T)

        variables = {}
        locations = {}
        for i in range(self.dim):
            for j in range(self.dim):
                var = f"assign_{i}_{j}"
                variables[var] = (0, T - 1)
                locations[var] = (i, j)
                self.varnames[(i,j)] = var
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

        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
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
        self.bdd_node = generator
        self.bdd = generator.bdd
        self.context = context

        self.is_compiled = True
        return end_time

    def get_assignment(self, sample, sample_format=SampleFormat.Value):
        num_bits = (self.bdd._number_of_cudd_vars() / self.dim) / self.dim
        final_assignment = []
        sample_bit_vec = []
        for bit in self.sample_as_bit_map(sample):
            if bit not in sample:
                sample_bit_vec.append(0)
            else:
                sample_bit_vec.append(sample[bit])

        if sample_format == SampleFormat.Bit:
            return sample_bit_vec

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

    def draw_simple_tiled_asp(self, sample_assignment):
        # print("sample_assignment: ", sample_assignment)
        # this assumes an order which it should not
        final_image_height = self.dim * self.tile_size[0]
        final_image_width = self.dim * self.tile_size[1]
        final_image = Image.new("RGBA", (final_image_height, final_image_width))

        self.tile_vec.sort(key=lambda x: x["index"])

        new_images = []

        for row in sample_assignment:
            # for el in row:
                # TODO: this is not the index you are looking for
                # el is the index of the tile type not the bit value?
                # tile_info = self.tile_vec[el]
            # print("row: ", row)
            tile_info = self.tile_vec[sample_assignment[row]]
            image_path = tile_info["image_path"]
            transform = tile_info["transformation"]
            new_images.append(self.processed_tile(image_path, transform))

        outer_index = 0
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                relvant_image = new_images[outer_index]
                final_image.paste(
                    relvant_image, ((i * self.tile_size[0]), (j * self.tile_size[1]))
                )
                outer_index += 1

        return final_image


    def draw_simple_tiled(self, sample_assignment):
        # this assumes an order which it should not
        final_image_height = self.dim * self.tile_size[0]
        final_image_width = self.dim * self.tile_size[1]
        final_image = Image.new("RGBA", (final_image_height, final_image_width))

        self.tile_vec.sort(key=lambda x: x["index"])



        # sample_assignment = np.transpose(sample_assignment)
        new_images = []
        for row in sample_assignment:
            for el in row:
                # TODO: this is not the index you are looking for
                # el is the index of the tile type not the bit value?
                tile_info = self.tile_vec[el] 
                # print(el)
                # print(row)
                # tile_info = self.tile_vec[row]
                image_path = tile_info["image_path"]
                transform = tile_info["transformation"]
                new_images.append(self.processed_tile(image_path, transform))

        outer_index = 0
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                relvant_image = new_images[outer_index]
                final_image.paste(
                    relvant_image, ((i * self.tile_size[0]), (j * self.tile_size[1]))
                )
                outer_index += 1

        return final_image

    def processed_tile_asp(self, image_path, transformation):
        im = Image.open(image_path)
        newim = im
        if transformation == 1:
            newim = im.rotate(270)
        elif transformation == 2:
            newim = im.rotate(180)
        elif transformation == 3:
            newim = im.rotate(90)
        elif transformation == 4:
            newim = im.mirror()
        elif transformation == 5:
            newim = newim.mirror()
            newim = im.rotate(90)
        elif transformation == 5:
            newim = newim.mirror()
            newim = im.rotate(270)
        elif transformation == 7:
            newim = newim.mirror()
            newim = im.rotate(270)
        elif transformation > 7:
            print(
                "transformation is invalid; not doing any transformation is the default"
            )

        return newim

    def processed_tile(self, image_path, transformation):
        im = Image.open(image_path)
        newim = im
        if transformation == 1:
            newim = im.rotate(90)
        elif transformation == 2:
            newim = im.rotate(180)
        elif transformation == 3:
            newim = im.rotate(270)
        elif transformation == 4:
            newim = im.mirror()
        elif transformation == 5:
            newim = newim.mirror()
            newim = im.rotate(90)
        elif transformation == 5:
            newim = newim.mirror()
            newim = im.rotate(180)
        elif transformation == 7:
            newim = newim.mirror()
            newim = im.rotate(270)
        elif transformation > 7:
            print(
                "transformation is invalid; not doing any transformation is the default"
            )

        return newim

    def get_spec_headers(self):
        return ["Dimension", "TileSet"]

    def get_header_value(self, header):
        if header == "Dimension":
            return self.dim
        elif header == "TileSet":
            return self.tileset.value

    def spec_string(self):
        return f"{self.name}-{self.dim}"

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
            for j in range(kop["dom"][1]):
                assignment = {str(assign_cell): j}
                rs = self.context.let(assignment, bdd_node)
                co = rs.count()
                assign_cell_count.append(co)
            assign_count[assign_cell] = assign_cell_count

            # print(assign_cell)
        return assign_count

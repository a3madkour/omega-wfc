from PIL import Image

def processed_tile(image_path, transformation):
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
        print("transformation is invalid; not doing any transformation is the default")

    return newim


 


def draw_simple_tiled(sample_assignment, tile_vec, dim, tile_size):
    # this assumes an order which it should not
    final_image_height = dim * tile_size[0]
    final_image_width = dim * tile_size[1]
    final_image = Image.new("RGBA", (final_image_height, final_image_width))

    tile_vec.sort(key=lambda x: x["index"])
    new_images = []
    for row in sample_assignment:
        for el in row:
            # TODO: this is not the index you are looking for
            # el is the index of the tile type not the bit value?
            tile_info = tile_vec[el]
            image_path = tile_info["image_path"]
            transform = tile_info["transformation"]
            new_images.append(processed_tile(image_path, transform))

    outer_index = 0
    for i in range(0, dim):
        for j in range(0, dim):
            relvant_image = new_images[outer_index]
            # TODO remove the below debug statement
            # relvant_image.save(f"{i}-{j}.png")
            final_image.paste(relvant_image, ((i * tile_size[0]), (j * tile_size[1])))
            outer_index += 1
    return final_image

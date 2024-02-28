import pandas as pd
from utils import get_all_elements_in_dir


# data_dir = "train-on-2207-models-10000-samples/Knots-2"
data_dir = "mario-1000-samples/Platformer"

for el in get_all_elements_in_dir(data_dir):
    data = pd.read_csv(f"{data_dir}/{el}", dtype=str)
    print(data)
    bit_strings = data["BitString"]
    unique_bit_strings = bit_strings.unique()
    count_bit_strings = bit_strings.value_counts()
    ax = count_bit_strings.plot.bar()
    ax.get_figure().savefig("nohello.png")
    print(count_bit_strings)
    break


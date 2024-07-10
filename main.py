from Experiment import Experiment, draw_simple_tiled_heat_map,Metric, draw_analytical_era_simpletiled
from Experiment import asp_run_test,draw_simple_tiled_samples,draw_asp_tilemap
from SimpleTiled import SimpleTiled,TileSet
import numpy as np






# draw_asp_tilemap("Knots", 2, 1)

generator = SimpleTiled(dim=7,tileset=TileSet.Knots)

print("compiling")
generator.compile()
print("done compiling")
generator
(
    compute_true_probs_time,
    sample_time,
    final_sample,
    sample_marginal
) = generator.sample(clear_true_probs=False)

for i in range(10):
    generator
    (
        compute_true_probs_time,
        sample_time,
        final_sample,
        sample_marginal
    ) = generator.sample(clear_true_probs=False)
    bmp = generator.sample_as_bit_map(final_sample)
    # print("bmp: ", bmp)
    assignment = generator.get_assignment(bmp)
    # print(assignment)
    img = generator.draw_simple_tiled(assignment)
    img.save(f"samples/example-sample{i}.png")
# generator.

# experiment = Experiment()
# experiment.add_metric_to_all_trials(Metric.bit_string_metric(experiment.generator))
# experiment.run(num_trials=1, num_samples = 1)
# draw_simple_tiled_samples(experiment)

# draw_simple_tiled_heat_map(experiment)
# draw_analytical_era_simpletiled(experiment)

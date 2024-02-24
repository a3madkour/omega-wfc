from enum import Enum
from SimpleTiled import SimpleTiled


class Metric:
    def __init__(self, metric_func, metric_name):
        self.metric_name = metric_name
        self.metric_func = metric_func

    #TODO define hash function as the name
    #TODO define obj callable as the metric func


class Trial:
    def __init__(
        self,
        generator,
        num_samples,
        dim,
        metrics,
        recompile_per_trial,
        clear_true_probs,
    ):
        self.generator = generator
        self.num_samples = num_samples
        self.metrics = metrics
        self.recompile_per_trial = recompile_per_trial
        self.clear_true_probs = clear_true_probs
        self.metric_data = []
        # time_data = compile time, sample time, learning time, generating training set time
        self.time_data = {}

    def run(self):
        if self.recompile_per_trial or not self.generator.is_compiled:
            compile_time = self.generator.compile()
            if "compile_time" not in self.time_data:
                self.time_data["compile_time"] = [compile_time]
            else:
                self.time_data["compile_time"].append(compile_time)

            if "sample_time" not in self.time_data:
                self.time_data["sample_time"] = []

            if "compute_true_probs_time" not in self.time_data:
                self.time_data["compute_true_probs_time"] = []

            for i in range(0, self.num_samples):
                (
                    compute_true_probs_time,
                    sample_time,
                    final_sample,
                ) = self.generator.sample(clear_true_probs=self.clear_true_probs)
                self.time_data["compute_true_probs_time"].append(
                    compute_true_probs_time
                )
                self.time_data["sample_time"].append(sample_time)
                for metric in self.metrics:
                    if metric not in self.metrics_data:
                        self.metrics_data[metric] = []
                    self.metrics_data[metric].append(metric(final_sample))

    def __repr__(self):
        out_str = ""
        for el in self.__dict__:
            out_str = f"{out_str}{el}:{self.__dict__[el]}\n"

        return out_str


class Experiment:
    # abstract out the experiment type/ have it be in SimpleTiled
    def __init__(
        self, generator=SimpleTiled(), num_samples=10, dim=2, trials=[], metrics=[]
    ):
        self.generator = generator
        self.num_samples = num_samples
        self.dim = dim
        self.trials = trials
        self.metrics = metrics
        self.metric_data = {}

    def __repr__(self):
        out_str = ""
        for el in self.__dict__:
            out_str = f"{out_str}{el}:{self.__dict__[el]}\n"

        return out_str

    def run(
        self,
        num_trials=10,
        num_samples=10,
        dim=2,
        metrics=[],
        recompile_per_trial=False,
        clear_true_probs=True,
    ):
        for i in range(num_trials):
            trial = Trial(
                self.generator, num_samples, dim, metrics, recompile_per_trial, clear_true_probs
            )
            trial.run()
            self.trials.append(trial)
            # print("trial: ", trial)

    def to_csv(self, filename):
        # should return an array of the metric applied to the runs
        pass

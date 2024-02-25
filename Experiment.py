from enum import Enum
import uuid
from SimpleTiled import SimpleTiled


#figure out how to compute kl-divergence and jenson on samples
#var counts as metrics
#bit string
#haming distance from a given bit string


class Metric:
    def __init__(self, metric_func, metric_name):
        self.metric_name = metric_name
        self.metric_func = metric_func


    def __repr__(self):
        return self.metric_name

    def __hash__(self):
        return self.metric_name.__hash__()

    def __call__(self, *args):
        return self.metric_func(*args)

    def hamming_distance(bit_string_1,bit_string_2):
        #TODO: add proper distance
        return "this is wrong"
    def hamming_distance_metric(generator, bit_string):
        return Metric(lambda x: Metric.hamming_distance(x,bit_string),f"Hamming-{bit_string}")

    def bit_string_metric(generator):
        return Metric(lambda x: generator.sample_as_bit_string(x),"BitString")


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
        self.metrics_data = {}
        # time_data = compile time, sample time, learning time, generating training set time
        self.trial_id = uuid.uuid4()
        self.time_data = {}



    def add_metric(self,metric):
        self.metrics.append(metric)
        print(self.metrics)
        
    def run(self):
        if self.recompile_per_trial or not self.generator.is_compiled:
            compile_time = self.generator.compile()
            print("Compiling")
            if "compile_time" not in self.time_data:
                self.time_data["compile_time"] = [compile_time]
            else:
                self.time_data["compile_time"].append(compile_time)
            print(f"Done: Compiling {compile_time / 1e09}s")
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

    def to_csv(self,path):
        # self.num_samples = num_samples
        # self.metrics = metrics
        # self.recompile_per_trial = recompile_per_trial
        # self.clear_true_probs = clear_true_probs
        # self.metrics_data = {}
        # time_data = compile time, sample time, learning time, generating training set time
        # self.time_data = {}
        filename = f"{path}/{self.generator.name}-{self.generator.size}/trial-{self.trial_id}.csv"

        
        if "sample_time" not in self.time_data:
            print("Trial did not run yet")
            return

        data_str = "SampleTime"
        if "learning_time" in self.time_data:
            data_str += ",LearningTime"

        if "gen_train_time" in self.time_data:
            data_str += ",GenTrainTime"

        for metric in self.metrics:
            data_str += f",{metric}"

        data_str += "\n"

        for i in range(self.num_samples):
            sample_time_data = self.time_data["sample_time"][i]
            data_str += f"{sample_time_data}"

            if "learning_time" in self.time_data:
                learning_time_data = self.time_data["learning_time"][i]
                data_str += f"{learning_time_data}"

            if "gen_train_time" in self.time_data:
                gen_train_time_data = self.time_data["gen_train_time"][i]
                data_str += f"{gen_train_time_data}"


            data_str += f"{sample_time_data}"
            for metric in self.metrics:
                data_str += f", {self.metrics_data[metric]}"
            data_str += "\n"

        # print(data_str)

class Experiment:
    # abstract out the experiment type/ have it be in SimpleTiled
    def __init__(
        self, generator = None , num_samples=10, dim=2, trials=[], metrics=[]
    ):
        if generator == None:
            generator = SimpleTiled(dim=dim)
        self.generator = generator
        self.num_samples = num_samples
        self.dim = dim
        self.trials = trials
        self.metrics = metrics
        self.metrics_data = {}
        print(dim)

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
        clear_true_probs=False,
    ):
        #clearing trials always
        self.trials.clear() 

        if len(metrics) > 0:
            self.metrics = metrics

        for i in range(num_trials):
            trial = Trial(
                self.generator, num_samples, dim, self.metrics, recompile_per_trial, clear_true_probs
            )
            self.trials.append(trial)
        for t in self.trials:
            t.run()

    def to_csv(self, path):
        # should return an array of the metric applied to the runs
        # we should also return the trial info somewhere in a csv
        i = 0
        for trial in self.trials:
            i += 1
            output_str= trial.to_csv("")
        pass

    def add_metric_to_all_trials(self,metric):
        self.metrics.append(metric)

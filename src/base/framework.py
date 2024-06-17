import os
import sys
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

np.float = float
np.int = int

sys.path.append('.')

from src.base.config_store import ConfigurationStore
from src.base.configuration import Configuration, ComparisonParameter
from src.base.model import NNClassifier
from src.core.online_learning.cost_vector_adaptor import CostVectorAdaptor
from src.helper.helper import set_seed
from src.helper.tracer import CITracer, MetricTracer
import argparse

current_path = os.path.dirname(__file__)


def run(configuration: Configuration):
    print(datetime.now())
    ConfigurationStore.set_configuration(config)
    print(f'running with parameter setting: {configuration.get_parameters().to_dict()}')
    set_seed(configuration.seed)
    print(f'run with seed {configuration.seed}')
    print(f'stream length {configuration.dataset_meta.n_instance}')
    model = CostVectorAdaptor(config=configuration)

    # label
    labels_recorder = []

    prob_uniform_recorder = []
    prob_naive_recorder = []
    prob_single_recorder = []
    prob_ensemble_recorder = []
    predict_uniform_recorder = []
    predict_naive_recorder = []
    predict_single_recorder = []
    predict_ensemble_recorder = []
    vector_naive_recorder = []
    vector_single_recorder = []
    vector_ensemble_recorder = []
    vectors_ensemble_recorder = []

    naive_gmean_recorder = []
    single_gmean_recorder = []
    ensemble_gmean_recorder = []
    naive_balanced_accuracy_recorder = []
    single_balanced_accuracy_recorder = []
    ensemble_balanced_accuracy_recorder = []
    naive_recall_recorder = []
    single_recall_recorder = []
    ensemble_recall_recorder = []

    std_recorder = []

    ci_tracer = CITracer(configuration)
    metric_tracer_naive = MetricTracer(configuration.dataset_meta.n_class)
    metric_tracer_single = MetricTracer(configuration.dataset_meta.n_class)
    metric_tracer_ensemble = MetricTracer(configuration.dataset_meta.n_class)

    data = configuration.dataset_meta.data
    labels = configuration.dataset_meta.labels
    data_label_list = list(enumerate(zip(data, labels), start=1))
    bar = tqdm(data_label_list)

    n_pretrain = int(configuration.pretrain_ratio * configuration.dataset_meta.n_instance)
    for epoch, (datum, label) in bar:
        if configuration.pretrain_ratio > 0 and epoch <= n_pretrain:
            ci_tracer.update(label.item())
            if epoch == n_pretrain:
                priors = ci_tracer.cost_candidates()
                model.step_batch(data[:n_pretrain], labels[:n_pretrain], configuration.train_validate_split, priors,
                                 evolving=ea_evolving, max_generation=max_init_generation)
            continue

        priors = ci_tracer.cost_candidates()

        evolving = (((epoch % configuration.evaluation_meta.evolution_f == 0) and ea_evolving)
                    or configuration.optimization_method == 'no')
        (prob_uniform, prob_naive, prob_single, prob_ensemble,
         predict_uniform, predict_naive, predict_single, predict_ensemble,
         vector_naive, vector_single, vector_ensemble, vectors_ensemble) = model.step(
            datum=datum, label=label, evolving=evolving, priors=priors)

        ci_tracer.update(label)
        metric_tracer_naive.update(predict_naive, label)
        metric_tracer_single.update(predict_single, label)
        metric_tracer_ensemble.update(predict_ensemble, label)

        labels_recorder.append(label.item())
        prob_uniform_recorder.append(prob_uniform)
        prob_naive_recorder.append(prob_naive)
        prob_single_recorder.append(prob_single)
        prob_ensemble_recorder.append(prob_ensemble)
        predict_uniform_recorder.append(predict_uniform)
        predict_naive_recorder.append(predict_naive)
        predict_single_recorder.append(predict_single)
        predict_ensemble_recorder.append(predict_ensemble)
        vector_naive_recorder.append(vector_naive)
        vector_single_recorder.append(vector_single)
        vector_ensemble_recorder.append(vector_ensemble)
        vectors_ensemble_recorder.append(vectors_ensemble)

        gmean_naive_tmp, balanced_accuracy_naive_tmp, recall_naive_tmp = metric_tracer_naive.instantaneous_metric()
        gmean_single_tmp, balanced_accuracy_single_tmp, recall_single_tmp = metric_tracer_single.instantaneous_metric()
        gmean_ensemble_tmp, balanced_accuracy_ensemble_tmp, recall_ensemble_tmp = metric_tracer_ensemble.instantaneous_metric()

        naive_gmean_recorder.append(gmean_naive_tmp)
        single_gmean_recorder.append(gmean_single_tmp)
        ensemble_gmean_recorder.append(gmean_ensemble_tmp)

        naive_balanced_accuracy_recorder.append(balanced_accuracy_naive_tmp)
        single_balanced_accuracy_recorder.append(balanced_accuracy_single_tmp)
        ensemble_balanced_accuracy_recorder.append(balanced_accuracy_ensemble_tmp)

        naive_recall_recorder.append(recall_naive_tmp)
        single_recall_recorder.append(recall_single_tmp)
        ensemble_recall_recorder.append(recall_ensemble_tmp)

        std_recorder.append(model.optimization.std)
        bar.set_description(
            f'decay gmean: {gmean_single_tmp:.3f}, decay balanced accuracy: {balanced_accuracy_single_tmp}')
    print('average gmean:', sum(single_gmean_recorder) / len(single_gmean_recorder))
    print('average balanced accuracy:', sum(single_balanced_accuracy_recorder) / len(single_balanced_accuracy_recorder))


def read_all():
    return {
        "elec": "Elec",
        "INSECTS-abrupt_balanced_norm": "Abrupt",
        "INSECTS-gradual_balanced_norm": "Gradual",
        "INSECTS-incremental_balanced_norm": "Incremental1",
        "luxembourg": "Luxembourg",
        "NOAA": "NOAA",
        "ozone": "Ozone",
        "airlines": "Airlines",
        "covtype": "Covtype",
        "INSECTS-incremental_imbalanced_norm": "Incremental2",
        "abalone-17_vs_7-8-9-10": "Abalone1",
        "abalone-19_vs_10-11-12-13": "Abalone2",
        "car-good": "Car1",
        "car-vgood": "Car2",
        "kddcup-rootkit-imap_vs_back": "Kddcup",
        "kr-vs-k-zero-one_vs_draw": "Kr",
        "segment0": "Segment",
        "shuttle-2_vs_5": "Shuttle1",
        "shuttle-c0-vs-c4": "Shuttle2",
        "thyroid": "Thyroid",
        "winequality-red-3_vs_5": "Win1",
        "winequality-red-4": "Win2",
        "winequality-red-8_vs_6": "Win3",
        "winequality-white-3-9_vs_5": "Win4",
        "winequality-white-3_vs_7": "Win5",
        "yeast-1-2-8-9_vs_7": "Yeast1",
        "yeast": "Yeast2",
        "yeast3": "Yeast3",
        "yeast5": "Yeast4",
        "yeast6": "Yeast5"
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=200)
    parser.add_argument("--dataset", type=str, default='elec')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oversampling_rate", type=int, default=5)
    parser.add_argument("--n_individual", type=int, default=25)
    parser.add_argument("--n_prior_individual", type=int, default=25)
    args = parser.parse_args()

    base_path = f'{current_path}/../log/{time.time():.0f}'
    config = Configuration(
        ComparisonParameter(
            ensemble_size=1, buffer_size=args.buffer_size, classifier=NNClassifier, dataset=args.dataset,
            seed=args.seed, optimization_method='soea'
        )
    )
    # default setting
    ea_evolving = True
    max_init_generation = 10
    config.soea_meta.n_generation = 1
    config.evaluation_meta.evolution_f = 5

    config.oversampling_rate = args.oversampling_rate
    config.soea_meta.n_individual = args.n_individual
    config.soea_meta.n_prior_individual = args.n_prior_individual
    run(config)

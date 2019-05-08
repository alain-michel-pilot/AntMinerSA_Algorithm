import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from dataset import Dataset
from ant_miner import AntMinerSA
from k_fold_crossvalidation import k_fold


def main_surval():

    # INPUT: USER-DEFINED PARAMETERS:
    no_of_ants = 3000
    min_cases_per_rule = 10
    max_uncovered_cases = 10
    no_rules_converg = 10

    # ANT-MINER ALGORITHM: list of rules generator
    ant_miner = AntMinerSA(no_of_ants, min_cases_per_rule, max_uncovered_cases, no_rules_converg)
    ant_miner.read_data()
    ant_miner.fit()

    # RUN INFORMATION LOG
    # f = open(log_file, "a+")
    # f.write('\n\n> FOLD: ' + repr(fold))
    # f.write('\n\n=== Run Information ===')
    # f.write('\nTraining cases: ' + repr(len(kfold_training_cases)) + ' ' + repr(kfold_training_cases))
    # f.write('\nTest cases: ' + repr(len(kfold_test_cases)) + ' ' + repr(kfold_test_cases))
    # f.write('\n\n=== Discovered Model ===')
    # f.close()
    # print('\nRULES:\n')
    # for rule in ant_miner.discovered_rule_list:
    #     rule.print(class_attr)
    #     rule.print_txt(log_file, class_attr)
    # f = open(log_file, "a+")
    # f.write('\n\nNumber of discovered rules: ' + repr(len(ant_miner.discovered_rule_list)))
    # f.write('\nPredictive Accuracy: ' + repr(accuracy) + '\n')
    # f.close()
    #
    # print('\nPREDICTIVE ACCURACY MEAN', predictive_accuracy_mean)
    # print('\nPREDICTIVE ACCURACY STD', predictive_accuracy_std)
    # f = open(log_file, "a+")
    # f.write('\n\n\n\n===== EXECUTION INFORMATION =====')
    # f.write('\nPredictive accuracy (mean +- std): ' + repr(predictive_accuracy_mean) +
    #         ' +- ' + repr(predictive_accuracy_std))
    # f.write('\nAverage number of discovered rules: ' + repr(no_of_discovered_rules_average) + '\n\n\n\n')
    # f.close()

    return


def main():

    # INPUT: USER-DEFINED PARAMETERS:
    no_of_ants = 3000
    min_cases_per_rule = 10
    max_uncovered_cases = 10
    no_rules_converg = 10

    # INPUT: DATASET AND CLASS ATTRIBUTE NAME
    log_file = "log_execution.txt"
    header = list(pd.read_csv('datasets/car_header.txt', delimiter=','))
    data = pd.read_csv('datasets/car.data.txt', delimiter=',', header=None, names=header, index_col=False)
    class_attr = 'Class'

    # K-FOLD CROSS-VALIDATION SETTINGS
    k = 10
    training_folders, test_folders = k_fold(data, class_attr, n_splits=k, stratified=True)

    # GLOBAL VARIABLES
    predictive_accuracy = []
    no_of_discovered_rules = []

    # K ITERATIONS OF ANT-MINER ALGORITHM AND CLASSIFICATION TASK BASED ON GENERATED RULES:
    for fold in range(k):
        print('\nFOLD: ', fold)

        # CONSTRUCTING DATASET FOR K ITERATION OF K-FOLD CROSS VALIDATION
        kfold_test_cases = test_folders[fold]
        kfold_training_cases = training_folders[fold]
        training_data = data.drop(kfold_test_cases, axis=0).copy()
        test_data = data.drop(kfold_training_cases, axis=0).copy()

        # Objects: TRAINING AND TEST DATASETS
        training_dataset = Dataset(training_data, class_attr)
        test_dataset = Dataset(test_data, class_attr)

        # ANT-MINER ALGORITHM: list of rules generator
        ant_miner = AntMinerSA(training_dataset, no_of_ants, min_cases_per_rule, max_uncovered_cases, no_rules_converg)
        ant_miner.fit()
        no_of_discovered_rules.append(len(ant_miner.discovered_rule_list))

        # CLASSIFICATION OF NEW CASES
        test_dataset_real_classes = test_dataset.get_real_classes()
        test_dataset_predicted_classes = ant_miner.predict(test_dataset)

        # PREDICTIVE ACCURACY CALCULATION
        accuracy = accuracy_score(test_dataset_real_classes, test_dataset_predicted_classes)
        predictive_accuracy.append(accuracy)

        # RUN INFORMATION LOG
        f = open(log_file, "a+")
        f.write('\n\n> FOLD: ' + repr(fold))
        f.write('\n\n=== Run Information ===')
        f.write('\nTraining cases: ' + repr(len(kfold_training_cases)) + ' ' + repr(kfold_training_cases))
        f.write('\nTest cases: ' + repr(len(kfold_test_cases)) + ' ' + repr(kfold_test_cases))
        f.write('\n\n=== Discovered Model ===')
        f.close()
        print('\nRULES:\n')
        for rule in ant_miner.discovered_rule_list:
            rule.print(class_attr)
            rule.print_txt(log_file, class_attr)
        f = open(log_file, "a+")
        f.write('\n\nNumber of discovered rules: ' + repr(len(ant_miner.discovered_rule_list)))
        f.write('\nPredictive Accuracy: ' + repr(accuracy) + '\n')
        f.close()

    # PREDICTIVE ACCURACY OF K-FOLDS
    predictive_accuracy_mean = np.mean(predictive_accuracy)
    predictive_accuracy_std = np.std(predictive_accuracy)
    no_of_discovered_rules_average = np.mean(no_of_discovered_rules)

    print('\nPREDICTIVE ACCURACY MEAN', predictive_accuracy_mean)
    print('\nPREDICTIVE ACCURACY STD', predictive_accuracy_std)
    f = open(log_file, "a+")
    f.write('\n\n\n\n===== EXECUTION INFORMATION =====')
    f.write('\nPredictive accuracy (mean +- std): ' + repr(predictive_accuracy_mean) +
            ' +- ' + repr(predictive_accuracy_std))
    f.write('\nAverage number of discovered rules: ' + repr(no_of_discovered_rules_average) + '\n\n\n\n')
    f.close()

    return


if __name__ == '__main__':
    # main()
    main_surval()

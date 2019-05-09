from ant_miner import AntMinerSA


def main():

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


if __name__ == '__main__':
    main()

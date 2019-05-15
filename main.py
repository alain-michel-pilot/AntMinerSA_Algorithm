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
    ant_miner.save_results("log_run.txt")

    return


if __name__ == '__main__':
    main()

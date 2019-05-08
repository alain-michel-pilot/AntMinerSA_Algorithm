import copy
import pandas as pd

from user_inputs import UserInputs
from terms_manager import TermsManager
from rule import Rule
from dataset import Dataset


class AntMinerSA:

    def __init__(self, no_of_ants, min_case_per_rule, max_uncovered_cases, no_rules_converg):
        self.dataset = None
        self.no_of_ants = no_of_ants
        self.min_case_per_rule = min_case_per_rule
        self.max_uncovered_cases = max_uncovered_cases
        self.no_rules_converg = no_rules_converg
        self.discovered_rule_list = []

    def read_data(self, ignore=None):  # use id_col_name in preprocessing: unify non-unique cases

        header = list(pd.read_csv(UserInputs.header_path, delimiter=','))
        data = pd.read_csv(UserInputs.data_path, delimiter=',', header=None, names=header, index_col=False)
        data.reset_index()

        self.dataset = Dataset(data)

        return

    def fit(self):
        no_of_remaining_cases = len(self.dataset.data)

        while no_of_remaining_cases > self.max_uncovered_cases:
            previous_rule = Rule(self.dataset)
            best_rule = copy.deepcopy(previous_rule)

            ant_index = 0
            converg_quality_index = 0
            converg_test_index = 1

            terms_mgr = TermsManager(self.dataset, self.min_case_per_rule)
            if not terms_mgr.available():
                break

            while True:
                if ant_index >= self.no_of_ants or converg_test_index >= self.no_rules_converg or converg_quality_index >= self.no_rules_converg:
                    break

                current_rule = Rule(self.dataset)
                current_rule.construct(terms_mgr, self.min_case_per_rule)
                current_rule.prune(terms_mgr)

                if current_rule.quality == 0.0:
                    converg_quality_index += 1
                elif current_rule.equals(previous_rule):
                    converg_test_index += 1
                else:
                    converg_test_index = 1
                    converg_quality_index = 1
                    if current_rule.quality > best_rule.quality:
                        best_rule = copy.deepcopy(current_rule)

                terms_mgr.pheromone_updating(current_rule.antecedent, current_rule.quality)
                previous_rule = copy.deepcopy(current_rule)
                ant_index += 1

            if best_rule.quality == 0.0:
                break
            else:
                self.discovered_rule_list.append(best_rule)
                self.dataset.data_updating(best_rule.covered_cases)
                no_of_remaining_cases = len(self.dataset.data)
        # END OF WHILE (AVAILABLE_CASES > MAX_UNCOVERED_CASES)

        # generating rule for remaining cases
        general_rule = Rule(self.dataset)
        general_rule.general_rule()
        self.discovered_rule_list.append(general_rule)

        return

    def predict(self, test_dataset):

        predicted_classes = []
        all_cases = len(test_dataset.data)

        rules = copy.deepcopy(self.discovered_rule_list[:-1])
        remaining_cases_rule = copy.deepcopy(self.discovered_rule_list[-1])

        for case in range(all_cases):  # for each new case
            chosen_class = None

            for rule in rules:  # sequential rule compatibility test
                compatibility = True
                for attr, value in rule.antecedent.items():
                    if value != test_dataset.data[case, test_dataset.col_index[attr]]:
                        compatibility = False
                        break
                if compatibility:
                    chosen_class = rule.consequent
                    break

            if chosen_class is None:
                chosen_class = remaining_cases_rule.consequent

            predicted_classes.append(chosen_class)

        return predicted_classes

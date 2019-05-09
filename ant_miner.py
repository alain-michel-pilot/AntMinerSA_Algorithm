import copy
import pandas as pd

from user_inputs import UserInputs
from terms_manager import TermsManager
from rule import Rule
from dataset import Dataset


class AntMinerSA:

    def __init__(self, no_of_ants, min_case_per_rule, max_uncovered_cases, no_rules_converg):
        self.no_of_ants = no_of_ants
        self.min_case_per_rule = min_case_per_rule
        self.max_uncovered_cases = max_uncovered_cases
        self.no_rules_converg = no_rules_converg

        self.discovered_rule_list = []
        self._dataset = None
        self._terms_manager = None
        self._no_of_uncovered_cases = None

    def read_data(self,
                  data_path=UserInputs.data_path,
                  header_path=UserInputs.header_path,
                  attr_survival_name=UserInputs.attr_survival_name,
                  attr_event_name=UserInputs.attr_event_name,
                  attr_id_name=UserInputs.attr_id_name,
                  attr_to_ignore=UserInputs.attr_to_ignore):  # use attr_id_name in preprocessing: unify non-unique cases

        header = list(pd.read_csv(header_path, delimiter=','))
        data = pd.read_csv(data_path, delimiter=',', header=None, names=header, index_col=False)
        data.reset_index()
        self._dataset = Dataset(data, attr_survival_name, attr_event_name, attr_id_name, attr_to_ignore)

        return

    def fit(self):
        # Initialization
        self._terms_manager = TermsManager(self._dataset, self.min_case_per_rule)
        self._no_of_uncovered_cases = self._dataset.get_no_of_uncovered_cases()

        while self._no_of_uncovered_cases > self.max_uncovered_cases:

            # local variables
            ant_index = 0
            converg_quality_index = 0
            converg_test_index = 1

            # Initialize rules
            previous_rule = Rule(self._dataset)
            best_rule = copy.deepcopy(previous_rule)

            while True:
                if ant_index >= self.no_of_ants or converg_test_index >= self.no_rules_converg or converg_quality_index >= self.no_rules_converg:
                    break

                current_rule = Rule(self._dataset)
                current_rule.construct(self._terms_manager, self.min_case_per_rule)
                current_rule.prune(self._terms_manager)

                if current_rule.quality == 0.0:
                    converg_quality_index += 1
                elif current_rule.equals(previous_rule):
                    converg_test_index += 1
                else:
                    converg_test_index = 1
                    converg_quality_index = 1
                    if current_rule.quality > best_rule.quality:
                        best_rule = copy.deepcopy(current_rule)

                self._terms_manager.pheromone_updating(current_rule.antecedent, current_rule.quality)
                previous_rule = copy.deepcopy(current_rule)
                ant_index += 1

            if best_rule.quality == 0.0:
                break
            else:
                self.discovered_rule_list.append(best_rule)
                self._dataset.update_covered_cases(best_rule.covered_cases)
                self._no_of_uncovered_cases = self._dataset.get_no_of_uncovered_cases()
        # END OF WHILE (AVAILABLE_CASES > MAX_UNCOVERED_CASES)

        # generating rule for remaining cases
        general_rule = Rule(self._dataset)
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

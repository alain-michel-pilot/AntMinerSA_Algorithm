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
        self._Dataset = None
        self._TermsManager = None
        self._no_of_uncovered_cases = None

    def _global_stopping_condition(self):
        if self._no_of_uncovered_cases < self.max_uncovered_cases:
            return True
        return False

    def _local_stopping_condition(self, ant_index, converg_test_index, converg_quality_index):
        if ant_index >= self.no_of_ants:
            return True
        elif converg_test_index >= self.no_rules_converg:
            return True
        elif converg_quality_index >= self.no_rules_converg:
            return True
        return False

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
        self._Dataset = Dataset(data, attr_survival_name, attr_event_name, attr_id_name, attr_to_ignore)

        return

    def fit(self):
        # Initialization
        self._TermsManager = TermsManager(self._Dataset, self.min_case_per_rule)
        self._no_of_uncovered_cases = self._Dataset.get_no_of_uncovered_cases()

        while not self._global_stopping_condition():

            # local variables
            ant_index = 0
            converg_quality_index = 0
            converg_test_index = 1

            # Initialize rules
            previous_rule = Rule(self._Dataset)
            best_rule = copy.deepcopy(previous_rule)

            while not self._local_stopping_condition(ant_index, converg_test_index, converg_quality_index):

                current_rule = Rule(self._Dataset)
                current_rule.construct(self._TermsManager, self.min_case_per_rule)
                current_rule.prune(self._TermsManager)

                if current_rule.quality == 0.0:
                    converg_quality_index += 1
                elif current_rule.equals(previous_rule):
                    converg_test_index += 1
                else:
                    converg_test_index = 1
                    converg_quality_index = 1
                    if current_rule.quality > best_rule.quality:
                        best_rule = copy.deepcopy(current_rule)

                self._TermsManager.pheromone_updating(current_rule.antecedent, current_rule.quality)
                previous_rule = copy.deepcopy(current_rule)
                ant_index += 1

            if best_rule.quality == 0.0:
                break
            else:
                self.discovered_rule_list.append(best_rule)
                self._Dataset.update_covered_cases(best_rule.covered_cases)
                self._no_of_uncovered_cases = self._Dataset.get_no_of_uncovered_cases()
        # END OF WHILE (AVAILABLE_CASES > MAX_UNCOVERED_CASES)

        # generating rule for remaining cases
        general_rule = Rule(self._Dataset)
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

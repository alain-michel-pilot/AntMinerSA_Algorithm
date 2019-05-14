import copy
import pandas as pd

from user_inputs import UserInputs
from terms_manager import TermsManager
from rule import Rule
from dataset import Dataset
from pruner import Pruner


class AntMinerSA:

    def __init__(self, no_of_ants, min_case_per_rule, max_uncovered_cases, no_rules_converg):
        self.no_of_ants = no_of_ants
        self.min_case_per_rule = min_case_per_rule
        self.max_uncovered_cases = max_uncovered_cases
        self.no_rules_converg = no_rules_converg

        self.discovered_rule_list = []
        self._Dataset = None
        self._TermsManager = None
        self._Pruner = None
        self._no_of_uncovered_cases = None

    def _global_stopping_condition(self, iterations):
        if self._no_of_uncovered_cases < self.max_uncovered_cases:
            return True
        if iterations >= self.no_of_ants:
            return True
        return False

    def _local_stopping_condition(self, ant_index, converg_test_index):
        if ant_index >= self.no_of_ants:
            return True
        elif converg_test_index >= self.no_rules_converg:
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
        self._Pruner = Pruner(self._Dataset, self._TermsManager)
        self._no_of_uncovered_cases = self._Dataset.get_no_of_uncovered_cases()

        iterations = 0
        while not self._global_stopping_condition(iterations):

            # local variables
            ant_index = 0
            converg_test_index = 1

            # Initialize rules
            previous_rule = Rule(self._Dataset)
            best_rule = copy.deepcopy(previous_rule)
            best_rule.quality = 1 - UserInputs.alpha

            while not self._local_stopping_condition(ant_index, converg_test_index):

                current_rule = Rule(self._Dataset)
                current_rule.construct(self._TermsManager, self.min_case_per_rule)
                current_rule = self._Pruner.prune(current_rule)

                if current_rule.equals(previous_rule):
                    converg_test_index += 1
                else:
                    converg_test_index = 1
                    if current_rule.quality > best_rule.quality:
                        best_rule = copy.deepcopy(current_rule)

                self._TermsManager.pheromone_updating(current_rule.antecedent, current_rule.quality)
                previous_rule = copy.deepcopy(current_rule)
                ant_index += 1

            if best_rule.quality == 1 - UserInputs.alpha:   # did not generate any rules
                break
            else:
                self.discovered_rule_list.append(best_rule)
                self._Dataset.update_covered_cases(best_rule.sub_group_cases)
                self._no_of_uncovered_cases = self._Dataset.get_no_of_uncovered_cases()
            iterations += 1
        # END OF WHILE (AVAILABLE_CASES > MAX_UNCOVERED_CASES)

        return

    def results(self, log_file):
        f = open(log_file, "a+")

        f.write('\n\n====== ANT-MINER PARAMETERS ======')
        f.write('\nNumber of ants: ' + repr(self.no_of_ants))
        f.write('\nNumber of minimum cases per rule: ' + repr(self.min_case_per_rule))
        f.write('\nNumber of maximum uncovered cases: ' + repr(self.max_uncovered_cases))
        f.write('\nNumber of rules for convergence: ' + repr(self.no_rules_converg))
        f.write('\n\n====== USER INPUTS PARAMETERS ======')
        f.write('\nHeuristic method: ' + repr(UserInputs.heuristic_method))
        f.write('\nAlpha value for KM function confidence interval: ' + repr(UserInputs.kmf_alpha))
        f.write('\nAlpha value for LogRank confidence: ' + repr(UserInputs.alpha))
        f.write('\n\n====== RUN INFORMATION ======')
        f.write('\nDatabase path: ' + repr(UserInputs.data_path))
        f.write('\nInstances: ' + repr(self._Dataset.data.shape[0]))
        f.write('\nAttributes: ' + repr(self._Dataset.data.shape[1]))
        f.write('\nNumber of remaining uncovered cases: ' + repr(self._no_of_uncovered_cases))
        f.write('\n\n====== DISCOVERED RULES ======')
        f.write('\n- Average survival on dataset: ' + repr(self._Dataset.average_survival) + '\n')
        f.close()

        for index, rule in enumerate(self.discovered_rule_list):
            rule.print_rule(log_file, index)

        return

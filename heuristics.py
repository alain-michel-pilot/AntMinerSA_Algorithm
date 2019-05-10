import math
import numpy as np
from user_inputs import UserInputs


class Heuristics:

    def __init__(self, dataset, heuristic_method=UserInputs.heuristic_method):
        self._Dataset = dataset
        self.heuristic_method = heuristic_method
        self._methods = ['attribute_stratification',
                         'attribute_value_stratification',
                         'survival_average_based_entropy']
        self._default_method = 'survival_average_based_entropy'

    def get_heuristic(self, attribute, value):

        if self.heuristic_method not in self._methods:
            print('Error: The heuristic method provided does not exist.'
                  'The survival_average_based_entropy method was used by default')
            self.heuristic_method = self._default_method

        if self.heuristic_method == 'attribute_stratification':
            return self.attribute_stratification(attribute, value)

        elif self.heuristic_method == 'attribute_value_stratification':
            return self.attribute_value_stratification(attribute, value)

        elif self.heuristic_method == 'survival_average_based_entropy':
            return self.survival_average_based_entropy(attribute, value)

        return

    def survival_average_based_entropy(self, attribute, value):
        # 2 classes: 0 for survival time < average_survival, 1 otherwise

        survival_average = self._Dataset.survival_times[1].mean()
        covered_rows = list(np.asarray(self._Dataset.data[:, self._Dataset.get_col_index(attribute)] == value).nonzero()[0])
        term_freq = len(covered_rows)

        class_freq = {}.fromkeys([0, 1], 0)
        for row in covered_rows:
            if self._Dataset.survival_times[1].iloc[row] < survival_average:
                class_freq[0] += 1
            else:
                class_freq[1] += 1

        # A POSTERIORI PROBABILITY: P(W|A=V) / ENTROPY = -SUM_FOR_ALL_CLASSES[ P(W|A=V) * log2(P(W|A=V)) ]
        entropy = 0
        for w in class_freq:
            prob_posteriori = class_freq[w] / term_freq
            if prob_posteriori != 0:
                entropy -= prob_posteriori * math.log2(prob_posteriori)

        return 1 - entropy

    def attribute_stratification(self, attribute, value):
        return  # heuristic

    def attribute_value_stratification(self, attribute, value):
        return  # heuristic


























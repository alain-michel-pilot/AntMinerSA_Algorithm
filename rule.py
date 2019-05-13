import collections
import copy
from user_inputs import UserInputs
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


class Rule:
    # RULE FORM:        IF < ANTECEDENT > THEN < CONSEQUENT >
    # ANTECEDENT FORM:  < Attribute_1 = Value > AND < Attribute_2 = Value > AND ... AND < Attribute_N = Value >
    # CONSEQUENT FORM:  Kaplan-Meier Survival Estimation

    def __init__(self, dataset):
        self.antecedent = {}
        self.consequent = {'survival_times': None, 'events': None, 'km_function': None}
        self.sub_group_cases = dataset.get_all_cases_index()
        self.no_covered_cases = len(self.sub_group_cases)
        self.quality = 0.0
        self.sub_group_complement = {'survival_times': None, 'events': None, 'km_function': None}
        self.logrank_test = None
        self._Dataset = dataset

    def _set_cases(self, cases):
        self.sub_group_cases = cases
        self.no_covered_cases = len(cases)
        return

    def _set_consequent(self):

        # Sub group induced by the rule
        self.consequent['survival_times'] = self._Dataset.survival_times[1].iloc[self.sub_group_cases]
        self.consequent['events'] = self._Dataset.events[1].iloc[self.sub_group_cases]

        # Complement of the induced sub group
        sg_complement = list(set(self.sub_group_cases) ^ set(self._Dataset.get_all_cases_index()))
        self.sub_group_complement['survival_times'] = self._Dataset.survival_times[1].iloc[sg_complement]
        self.sub_group_complement['events'] = self._Dataset.events[1].iloc[sg_complement]

        # Kaplan-Meier estimations for sub group and complement
        kmf = KaplanMeierFitter()

        kmf.fit(self.consequent['survival_times'], self.consequent['events'],
                label='KM estimates for discovered subgroup', alpha=UserInputs.kmf_alpha)
        self.consequent['km_function'] = copy.deepcopy(kmf)

        kmf.fit(self.sub_group_complement['survival_times'], self.sub_group_complement['events'],
                label='KM estimates for discovered subgroup complement', alpha=UserInputs.kmf_alpha)
        self.sub_group_complement['km_function'] = copy.deepcopy(kmf)

        return

    def _set_quality(self):

        self.logrank_test = logrank_test(self.consequent['survival_times'],
                                         self.sub_group_complement['survival_times'],
                                         self.consequent['events'],
                                         self.sub_group_complement['events'])
        self.quality = 1 - self.logrank_test.p_value

        return

    def construct(self, terms_mgr, min_case_per_rule):

        # ANTECEDENT CONSTRUCTION
        while terms_mgr.available():

            term = terms_mgr.sort_term()
            covered_cases = list(set(term.covered_cases) & set(self.sub_group_cases))

            if len(covered_cases) >= min_case_per_rule:
                self.antecedent[term.attribute] = term.value
                self._set_cases(covered_cases)
                terms_mgr.update_availability(term.attribute)
            else:
                break

        self._set_consequent()
        self._set_quality()

        return

    def equals(self, prev_rule):

        attr_this = list(self.antecedent.keys())
        attr_prev = list(prev_rule.antecedent.keys())

        if self.consequent == prev_rule.consequent:
            if len(set(attr_this) ^ set(attr_prev)) == 0:   # both have same keys
                for attr in attr_this:
                    if self.antecedent[attr] != prev_rule.antecedent[attr]:
                        return False
            else:
                return False
        else:
            return False

        return True

    def general_rule(self):

        class_col = self._Dataset.col_index[self._Dataset.class_attr]
        if len(self._Dataset.data) == 0:
            original_data = self._Dataset.get_original_data()
            classes = original_data[:, class_col]
        else:
            classes = self._Dataset.data[:, class_col]

        class_freq = dict(collections.Counter(classes))

        max_freq = 0
        chosen_class = None
        for w, freq in class_freq.items():  # other way: class_chosen <= max(class_freq[])
            if freq > max_freq:
                chosen_class = w
                max_freq = freq

        self.sub_group_cases = []
        self.no_covered_cases = None
        self.quality = None
        self.consequent = chosen_class

        return

    def print(self, class_attr):

        print("IF { ", end="")

        antecedent_attrs = list(self.antecedent.keys())
        qtd_of_terms = len(antecedent_attrs)

        for t in range(0, qtd_of_terms):
            print(antecedent_attrs[t] + " = " + str(self.antecedent[antecedent_attrs[t]]), end="")

            if t < qtd_of_terms - 1:
                print(" AND ", end="")

        print(" } THAN { " + class_attr + " = " + str(self.consequent) + " }")

        return

    def print_txt(self, file, class_attr):

        antecedent_attrs = list(self.antecedent.keys())
        qtd_of_terms = len(antecedent_attrs)

        f = open(file, "a+")
        f.write('\nIF ')
        for t in range(0, qtd_of_terms):
            f.write(repr(antecedent_attrs[t]) + ' = ' + repr(self.antecedent[antecedent_attrs[t]]))
            if t < qtd_of_terms - 1:
                f.write(' AND ')

        f.write(' THAN ' + repr(class_attr) + ' = ' + repr(self.consequent))
        f.close()

        return

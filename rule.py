import collections
import numpy as np


class Rule:

    def __init__(self, dataset):
        self.antecedent = {}
        self.consequent = None
        self.covered_cases = dataset.get_all_cases_index()
        self.no_covered_cases = len(self.covered_cases)
        self.quality = 0.0
        self._dataset = dataset

    def __set_consequent(self):

        class_idx = self.__dataset.col_index[self.__dataset.class_attr]
        covered_rows = []
        max_freq = 0
        class_chosen = None

        for row in self.covered_cases:
            covered_rows.append(self.__dataset.data[row])
        covered_rows = np.array(covered_rows)

        class_freq = dict(collections.Counter(covered_rows[:, class_idx]))
        for w in class_freq:
            if class_freq[w] > max_freq:
                class_chosen = w
                max_freq = class_freq[w]

        self.consequent = class_chosen

        return

    def __set_quality(self):

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        class_idx = self.__dataset.col_index[self.__dataset.class_attr]

        for row_idx in range(len(self.__dataset.data)):
            # positive cases (TP|FP): covered by the rule
            if row_idx in self.covered_cases:
                if self.__dataset.data[row_idx, class_idx] == self.consequent:
                    tp += 1
                else:  # covered but doesnt have the class predicted
                    fp += 1
            # negative cases (TN|FN): not covered by the rule
            else:
                if self.__dataset.data[row_idx, class_idx] == self.consequent:
                    fn += 1
                else:  # not covered and doesnt have the class predicted
                    tn += 1

        den1 = (tp + fn)
        den2 = (fp + tn)
        if den1 == 0:
            self.quality = 0.0
        elif den2 == 0:
            self.quality = 1.0
        else:
            self.quality = (tp / den1) * (tn / den2)

        return

    def construct(self, terms_mgr, min_case_per_rule):

        # ANTECEDENT CONSTRUCTION
        while terms_mgr.available():

            term = terms_mgr.sort_term()
            covered_cases = list(set(term.covered_cases) & set(self.covered_cases))

            if len(covered_cases) >= min_case_per_rule:
                self.antecedent[term.attribute] = term.value
                self.covered_cases = covered_cases
                self.no_covered_cases = len(self.covered_cases)
                terms_mgr.update_availability(term.attribute)
            else:
                break

        self.__set_consequent()
        self.__set_quality()

        return

    def prune(self, terms_mgr):

        while len(self.antecedent) > 1:
            # current rule attributes
            current_antecedent = self.antecedent.copy()
            current_consequent = self.consequent
            current_cases = self.covered_cases
            current_quality = self.quality

            # Iteratively removes one attribute from current antecedent
            best_attr = None
            best_quality = current_quality

            for attr in current_antecedent:
                # new rule attributes
                self.antecedent.pop(attr, None)
                self.covered_cases = terms_mgr.get_cases(self.antecedent)
                self.__set_consequent()
                self.__set_quality()

                if self.quality >= best_quality:
                    best_attr = attr
                    best_quality = self.quality

                # restore current rule attributes
                self.antecedent = current_antecedent.copy()
                self.consequent = current_consequent
                self.covered_cases = current_cases
                self.quality = current_quality

            if best_attr is None:
                break
            else:  # save best pruned rule
                self.antecedent.pop(best_attr, None)
                self.covered_cases = terms_mgr.get_cases(self.antecedent)
                self.no_covered_cases = len(self.covered_cases)
                self.__set_consequent()
                self.__set_quality()

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

        class_col = self.__dataset.col_index[self.__dataset.class_attr]
        if len(self.__dataset.data) == 0:
            original_data = self.__dataset.get_original_data()
            classes = original_data[:, class_col]
        else:
            classes = self.__dataset.data[:, class_col]

        class_freq = dict(collections.Counter(classes))

        max_freq = 0
        chosen_class = None
        for w, freq in class_freq.items():  # other way: class_chosen <= max(class_freq[])
            if freq > max_freq:
                chosen_class = w
                max_freq = freq

        self.covered_cases = []
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

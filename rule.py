import collections
import numpy as np


class Rule:

    def __init__(self, dataset):
        self.antecedent = {}
        self.consequent = None
        self.covered_cases = list(range(len(dataset.data)))
        self.no_covered_cases = len(dataset.data)
        self.quality = 0.0
        self.__dataset = dataset

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

    def __set_quality(self, idx_e, idx_i, p):

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

        # just for log register
        if self.quality == 1 or self.quality == 0 or self.quality == 0.0:
            q_log_file = "log_rule-quality-analisys.txt"
            f = open(q_log_file, "a+")
            f.write('\n\n\n==============================================================================================================')
            f.write('\n=============== RULE QUALITY ANALISYS ========================================================================')
            f.write('\n==============================================================================================================')
            f.write('\n\n=> Code reference:')
            f.write('\n- idx_e: ' + repr(idx_e))
            f.write('\n- idx_i: ' + repr(idx_i))
            f.write('\n- pruning: ' + repr(p))
            f.write('\n\n=> Quality calculation info:')
            f.write('\n- Quality: ' + repr(self.quality))
            f.write('\n- TP: ' + repr(tp))
            f.write('\n- FP: ' + repr(fp))
            f.write('\n- FN: ' + repr(fn))
            f.write('\n- TN: ' + repr(tn) + '\n')
            f.close()
            self.print_txt(q_log_file, 'Class')

            array_log_file = "log_rule-quality-analisys_array-e" + str(idx_e) + "i" + str(idx_i) + "p" + str(p) + ".txt"
            f = open(q_log_file, "a+")
            f.write('\n- covered cases: ' + repr(self.covered_cases))
            f.write('\n- number of covered cases: ' + repr(self.no_covered_cases))
            f.write('\n\n>> DATASET USED FOR CALCULATION: ' + repr(array_log_file) + ' file <=')
            f.close()
            np.savetxt(array_log_file, self.__dataset.data, fmt='%5s')

        return

    def construct(self, terms_mgr, min_case_per_rule, idx_e, idx_i):
        c_log_file = "log_rule-construct.txt"
        f = open(c_log_file, "a+")
        f.write('\n\n\n=============== RULE CONSTRUCTION LOOP ======================================================================')
        f.write('\n> Stopping condition: rule cover less than minimum cases or there is no more attributes to be added')
        f.write('\n> Sequential construction: Sort a term to be added to rule > check number of covered cases ')
        f.write('\n> IF Stopping condition: set rule consequent')
        f.write('\n=============================================================================================================')
        f.write('\n EXTERNAL LOOP ITERATION ' + repr(idx_e))
        f.write('\n INTERNAL LOOP ITERATION ' + repr(idx_i))
        f.close()
        idx = 0

        # ANTECEDENT CONSTRUCTION
        while True:
            idx += 1
            f = open(c_log_file, "a+")
            f.write('\n\n>>>>>>>>>>>>>>> ITERATION ' + repr(idx))
            f.close()
            f = open(c_log_file, "a+")
            f.write('\n> List_of_terms size: ' + repr(terms_mgr.size()))
            f.write('\n\n==> CURRENT RULE:')
            f.close()
            self.print_txt(c_log_file, 'Class')

            if not terms_mgr.available():
                f = open(c_log_file, "a+")
                f.write('\n\n=============== END CONSTRUCTION')
                f.write('\n> Condition: empty terms list')
                f.write('\n   - current_list_of_terms size = ' + repr(terms_mgr.size()))
                f.write('\n   - iteration number = ' + repr(idx))
                f.close()
                break

            term = terms_mgr.new_sort_term()
            if term is None:
                print("Error: sorted term is None")
            covered_cases = list(set(term.covered_cases) & set(self.covered_cases))

            f = open(c_log_file, "a+")
            f.write('\n\n==> TERM TO BE ADDED: Attribute='+repr(term.attribute)+' Value='+repr(term.value))
            f.write('\n- Current rule covered cases: ' + repr(self.covered_cases))
            f.write('\n- Sorted term covered cases: ' + repr(term.covered_cases))
            f.write('\n- Intersection covered cases: ' + repr(covered_cases))
            f.write('\n- Number of intersection covered cases: ' + repr(len(covered_cases)))
            f.close()

            if len(covered_cases) >= min_case_per_rule:
                self.antecedent[term.attribute] = term.value
                self.covered_cases = covered_cases
                self.no_covered_cases = len(self.covered_cases)
                terms_mgr.update_availability(term.attribute)

                f = open(c_log_file, "a+")
                f.write('\n\n==> NEW CONSTRUCTED RULE:')
                f.close()
                self.print_txt(c_log_file, 'Class')
            else:
                f = open(c_log_file, "a+")
                f.write('\n\n=============== END CONSTRUCTION')
                f.write('\n> Condition: new rule doesnt cover minimum cases')
                f.write('\n   - current_list_of_terms size = ' + repr(terms_mgr.size()))
                f.write('\n   - iteration number = ' + repr(idx))
                f.close()
                break

        # CONSEQUENT SELECTION
        self.__set_consequent()
        if self.consequent is None:
            print("!! Error: Rule created with no consequent")

        # SET QUALITY
        self.__set_quality(idx_e, idx_i, p=False)

        f = open(c_log_file, "a+")
        f.write('\n\n>>> FINAL RULE ')
        f.close()
        self.print_txt(c_log_file, 'Class')
        f = open(c_log_file, "a+")
        f.write('\n-no_covered_cases: ' + repr(self.no_covered_cases))
        f.write('\n-quality: ' + repr(self.quality))
        f.write('\n\n> Number of iterations: ' + repr(idx))
        f.close()

        return

    def prune(self, terms_mgr, idx_e, idx_i):
        p_log_file = "log_rule-pruning-fnc.txt"
        f = open(p_log_file, "a+")
        f.write('\n\n\n================== RULE PRUNING LOOP =========================================================================')
        f.write('\n> Stopping condition: pruned rule quality be less than best quality so far or if pruned rule antecedent has just one term')
        f.write('\n> Receives constructed rule > drops each term on antecedent, sequentially from first to last > each term dropped consists on another rule > new pruned rule is the one of higher quality ')
        f.write('\n> IF no new rules have higher quality than the new pruned rule, or if new pruned rule has oly one term in the antecedent > returns pruned rule')
        f.write('\n==============================================================================================================')
        f.write('\n EXTERNAL LOOP ITERATION ' + repr(idx_e))
        f.write('\n INTERNAL LOOP ITERATION ' + repr(idx_i))
        f.write('\n\n> RULE TO BE PRUNED :')
        f.close()
        self.print_txt(p_log_file, 'Class')
        f = open(p_log_file, "a+")
        f.write('\n-Number of covered cases: ' + repr(self.no_covered_cases))
        f.write('\n-Quality: ' + repr(self.quality))
        f.close()

        idx = 0
        while True:
            idx += 1
            f = open(p_log_file, "a+")
            f.write('\n\n>>>>>>>>>>>>>>> ITERATION ' + repr(idx))
            f.close()

            if len(self.antecedent) <= 1:
                f = open(p_log_file, "a+")
                f.write('\n\n==> BREAK LOOP:')
                f.write('\n> Condition: pruned rule antecedent = 1')
                f.close()
                break

            f = open(p_log_file, "a+")
            f.write('\n\n==> CURRENT RULE :')
            f.close()
            self.print_txt(p_log_file, 'Class')
            f = open(p_log_file, "a+")
            f.write('\n\n==> TERMS DROPPING PROCEDURE:')
            f.close()

            # current rule attributes
            current_antecedent = self.antecedent.copy()
            current_consequent = self.consequent
            current_cases = self.covered_cases
            current_quality = self.quality

            # Iteratively removes one attribute from current antecedent
            best_attr = None
            best_quality = current_quality
            for attr in current_antecedent:

                f = open(p_log_file, "a+")
                f.write('\n\n>>> TERM: ')
                f.write('\n\n> Term_2b_dropped: Attribute=' + repr(attr) + ' Value=' + repr(current_antecedent[attr]))
                f.close()

                # new rule attributes
                self.antecedent.pop(attr, None)
                self.covered_cases = terms_mgr.get_cases(self.antecedent)
                self.__set_consequent()
                self.__set_quality(idx_e, idx_i, p=True)

                f = open(p_log_file, "a+")
                f.write('\n\n> Pruned Rule:')
                f.close()
                self.print_txt(p_log_file, 'Class')
                f = open(p_log_file, "a+")
                f.write('\n-Number of covered cases: ' + repr(len(self.covered_cases)))
                f.write('\n-Quality: ' + repr(self.quality))
                f.close()

                if self.quality >= best_quality:
                    best_attr = attr
                    best_quality = self.quality
                    f = open(p_log_file, "a+")
                    f.write('\n\n!!! Improvement: Best attribute to be removed so far')
                    f.close()

                # restore current rule attributes
                self.antecedent = current_antecedent.copy()
                self.consequent = current_consequent
                self.covered_cases = current_cases
                self.quality = current_quality

            if best_attr is None:
                f = open(p_log_file, "a+")
                f.write('\n\n==> BREAK LOOP:')
                f.write('\n> Condition: best quality of new pruned rules < current rule quality')
                f.close()
                break
            else:  # save best pruned rule
                self.antecedent.pop(best_attr, None)
                self.covered_cases = terms_mgr.get_cases(self.antecedent)
                self.no_covered_cases = len(self.covered_cases)
                self.__set_consequent()
                self.__set_quality(idx_e, idx_i, p=True)

        f = open(p_log_file, "a+")
        f.write('\n\n================== END PRUNING FUNCTION LOOP')
        f.write('\n> Condition: ')
        f.write('\n  - Number of iterations: ' + repr(idx))
        f.write('\n\n> Final Pruned Rule:')
        f.close()
        self.print_txt(p_log_file, 'Class')

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

        if self.consequent is None:
            print("!! Error: Rule created with no consequent")

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
        f.write('\n*RULE:   IF { ')
        for t in range(0, qtd_of_terms):
            f.write(repr(antecedent_attrs[t]) + ' = ' + repr(self.antecedent[antecedent_attrs[t]]))
            if t < qtd_of_terms - 1:
                f.write(' AND ')

        f.write(' } THAN { ' + repr(class_attr) + ' = ' + repr(self.consequent) + ' }')
        f.close()

        return

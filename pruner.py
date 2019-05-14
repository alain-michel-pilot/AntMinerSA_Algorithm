import copy
from rule import Rule


class Pruner:

    def __init__(self, dataset, terms_mgr):
        self._terms_mgr = terms_mgr
        self._dataset = dataset
        self._pruning_flag = False
        self.current_rule = None

    def prune(self, rule):

        self.current_rule = copy.deepcopy(rule)

        while len(self.current_rule.antecedent) > 1:

            current_antecedent = self.current_rule.antecedent.copy()

            for attr in current_antecedent:
                # new pruned rule antecedent and cases
                pruned_rule = Rule(self._dataset)
                pruned_rule.antecedent = current_antecedent.copy()
                pruned_rule.antecedent.pop(attr, None)
                pruned_rule.set_cases(self._terms_mgr.get_cases(pruned_rule.antecedent))
                pruned_rule.set_sub_group()
                pruned_rule.set_quality()

                if pruned_rule.quality >= self.current_rule.quality:
                    self._pruning_flag = True
                    self.current_rule = copy.deepcopy(pruned_rule)

            if not self._pruning_flag:
                break

        return self.current_rule

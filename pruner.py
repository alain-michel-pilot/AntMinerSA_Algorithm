import copy
from rule import Rule
from user_inputs import UserInputs
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


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
                # new rule attributes
                pruned_rule = Rule(self._dataset)
                pruned_rule.antecedent = current_antecedent.copy()
                pruned_rule.antecedent.pop(attr, None)
                pruned_rule.sub_group_cases = self._terms_mgr.get_cases(pruned_rule.antecedent)
                pruned_rule.no_covered_cases = len(pruned_rule.sub_group_cases)

                # Consequent of the induced sub group
                pruned_rule.consequent['survival_times'] = self._dataset.survival_times[1].iloc[pruned_rule.sub_group_cases]
                pruned_rule.consequent['events'] = self._dataset.events[1].iloc[pruned_rule.sub_group_cases]
                # Complement of the induced sub group
                sg_complement = list(set(pruned_rule.sub_group_cases) ^ set(self._dataset.get_all_cases_index()))
                pruned_rule.sub_group_complement['survival_times'] = self._dataset.survival_times[1].iloc[sg_complement]
                pruned_rule.sub_group_complement['events'] = self._dataset.events[1].iloc[sg_complement]
                pruned_rule.quality, pruned_rule.logrank_test = self._get_quality(pruned_rule.consequent['survival_times'],
                                                                                  pruned_rule.sub_group_complement['survival_times'],
                                                                                  pruned_rule.consequent['events'],
                                                                                  pruned_rule.sub_group_complement['events'])

                if pruned_rule.quality >= self.current_rule.quality:
                    self._pruning_flag = True
                    pruned_rule.consequent['km_function'] = self._get_km_estimate(pruned_rule.consequent['survival_times'],
                                                                                  pruned_rule.consequent['events'])
                    pruned_rule.sub_group_complement['km_function'] = self._get_km_estimate(pruned_rule.consequent['survival_times'],
                                                                                            pruned_rule.consequent['events'],
                                                                                            sub_group=False)
                    self.current_rule = copy.deepcopy(pruned_rule)

            if not self._pruning_flag:
                break

        return self.current_rule

    @staticmethod
    def _get_quality(sg_survival_times, sg_complement_survival_times, sg_events, sg_complement_events):

        statistical_test = logrank_test(sg_survival_times,
                                        sg_complement_survival_times,
                                        sg_events,
                                        sg_complement_events)
        quality = 1 - statistical_test.p_value

        return quality, statistical_test

    @staticmethod
    def _get_km_estimate(survival_times, events, sub_group=True):
        kmf = KaplanMeierFitter()
        if sub_group:
            kmf.fit(survival_times, events,
                    label='KM estimates for discovered subgroup', alpha=UserInputs.kmf_alpha)
        else:
            kmf.fit(survival_times, events,
                    label='KM estimates for discovered subgroup complement', alpha=UserInputs.kmf_alpha)
        return kmf

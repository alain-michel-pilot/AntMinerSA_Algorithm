from user_inputs import UserInputs
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams


class Rule:
    # RULE FORM:        IF < ANTECEDENT > THEN < CONSEQUENT >
    # ANTECEDENT FORM:  < Attribute_1 = Value > AND < Attribute_2 = Value > AND ... AND < Attribute_N = Value >
    # CONSEQUENT FORM:  <survival_average> <quality> <logrank p-value>

    def __init__(self, dataset):
        self.antecedent = {}
        self.sub_group_cases = dataset.get_instances()
        self.no_covered_cases = len(self.sub_group_cases)
        self.sub_group = {'survival_times': None, 'events': None}
        self.sub_group_complement = {'survival_times': None, 'events': None}
        self.quality = 0.0
        self.logrank_test = None
        self.string_repr = ()
        self._Dataset = dataset

    def set_cases(self, cases):
        self.sub_group_cases = cases
        self.no_covered_cases = len(cases)
        return

    def set_sub_group(self):

        # Sub group induced by the rule
        self.sub_group['survival_times'] = self._Dataset.survival_times[1].iloc[self.sub_group_cases]
        self.sub_group['events'] = self._Dataset.events[1].iloc[self.sub_group_cases]

        # Complement of the induced sub group
        sg_complement = list(set(self.sub_group_cases) ^ set(self._Dataset.get_instances()))
        self.sub_group_complement['survival_times'] = self._Dataset.survival_times[1].iloc[sg_complement]
        self.sub_group_complement['events'] = self._Dataset.events[1].iloc[sg_complement]

        return

    def set_quality(self):

        self.logrank_test = logrank_test(self.sub_group['survival_times'],
                                         self.sub_group_complement['survival_times'],
                                         self.sub_group['events'],
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
                self.set_cases(covered_cases)
                terms_mgr.update_availability(term.attribute)
            else:
                break

        self.set_sub_group()
        self.set_quality()

        return

    def equals(self, prev_rule):

        antecedent_this = list(self.antecedent.keys())
        antecedent_prev = list(prev_rule.antecedent.keys())

        if len(set(antecedent_this) ^ set(antecedent_prev)) == 0:   # both have same keys
            for attr in antecedent_this:
                if self.antecedent[attr] != prev_rule.antecedent[attr]:
                    return False
        else:
            return False

        return True

    def set_string_repr(self, index):
        rule_id = 'R' + str(index)
        average_survival = self.sub_group['survival_times'].mean()
        quality = self.quality
        p_value = self.logrank_test.p_value

        string = 'IF <' +\
                 '> AND <'.join(['{} = {}'.format(key, value) for (key, value) in self.antecedent.items()]) +\
                 '> THAN <average_survival = {}> <quality = {}> <p_value = {}>'.format(average_survival, quality, p_value)

        self.string_repr = (rule_id, string)
        return

    def print_rule(self, file):

        f = open(file, "a+")
        f.write('\n')
        f.write(self.string_repr[1])
        f.close()

        print(self.string_repr[0] + ': ' + self.string_repr[1])

        return

    def plot_km_estimates(self, index):
        # Kaplan-Meier estimations for sub group and complement

        rcParams['figure.figsize'] = 15, 6
        plt.figure(index + 1)
        ax = plt.subplot(111)
        fig_id = 'R' + str(index) + '_model'

        kmf_sg = KaplanMeierFitter()
        kmf_cpl = KaplanMeierFitter()

        kmf_sg.fit(self.sub_group['survival_times'], self.sub_group['events'],
                   label='KM estimates for subgroup', alpha=UserInputs.kmf_alpha)
        kmf_sg.plot(ax=ax)

        kmf_cpl.fit(self.sub_group_complement['survival_times'], self.sub_group_complement['events'],
                    label='KM estimates for complement', alpha=UserInputs.kmf_alpha)
        kmf_cpl.plot(ax=ax)

        title = self.string_repr[0] + ': ' + self.string_repr[1]
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Survival probability')
        plt.savefig(fig_id)

        return

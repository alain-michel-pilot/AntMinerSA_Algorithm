import numpy as np
from term import Term


class TermsManager:

    def __init__(self, dataset, min_case_per_rule):
        self._terms = {}
        self._availability = {}
        self._attr_values = {}
        self._pheromone_table = {}
        self._heuristic_table = {}
        self._no_of_terms = 0

        # build object
        self._constructor(dataset, min_case_per_rule)

    def _constructor(self, dataset, min_case_per_rule):

        # Attrubute-Values from the entire dataset
        attr_values = dataset.attr_values.copy()
        attrs = list(attr_values.keys())

        heuristic_accum = 0

        # TABLES: { Attr : {} }
        self._terms = {}.fromkeys(attrs, {})
        self._pheromone_table = {}.fromkeys(attrs, {})
        self._heuristic_table = {}.fromkeys(attrs, {})

        # TERMS: constructing _terms_dict and _availability
        self._attr_values = {}.fromkeys(attrs, [])
        self._availability = {}.fromkeys(attrs)
        for attr, values in attr_values.items():
            list_of_values = []
            values_dict = {}.fromkeys(values)
            for value in values:
                term_obj = Term(attr, value, dataset, min_case_per_rule)
                if term_obj.available():
                    values_dict[value] = term_obj
                    list_of_values.append(value)
                    self._no_of_terms += 1
                    heuristic_accum += term_obj.get_heuristic()
                else:
                    values_dict.pop(value, None)

            if not list_of_values:
                self._terms.pop(attr)
                self._pheromone_table.pop(attr, None)
                self._heuristic_table.pop(attr, None)
                self._attr_values.pop(attr, None)
                self._availability.pop(attr, None)
            else:
                self._terms[attr] = values_dict
                self._attr_values[attr] = list_of_values[:]
                self._availability[attr] = True

        # No available terms on dataset
        if self._no_of_terms == 0:
            return

        # _pheromone_table: {Attr : {Value : Pheromone}} | _heuristic_table: {Attr : {Value : Heuristic}}
        initial_pheromone = 1 / self._no_of_terms
        for attr, values in self._attr_values.items():
            self._pheromone_table[attr] = {}.fromkeys(values, initial_pheromone)
            self._heuristic_table[attr] = {}.fromkeys(values)
            for value in values:
                self._heuristic_table[attr][value] = (self._terms[attr][value].get_heuristic() / heuristic_accum)

        return

    def _get_prob_accum(self):

        accum = 0
        for attr, values in self._attr_values.items():
            if self._availability[attr]:
                for value in values:
                    accum += self._heuristic_table[attr][value] * self._pheromone_table[attr][value]

        return accum

    def _get_pheromone_accum(self):

        accum = 0
        for attr, values in self._attr_values.items():
            for value in values:
                accum += self._pheromone_table[attr][value]

        return accum

    def _reset_availability(self):

        attrs = list(self._attr_values.keys())
        self._availability = {}.fromkeys(attrs, True)

        return

    def _get_probabilities(self):

        prob_accum = self._get_prob_accum()
        probabilities = []

        for attr, values in self._attr_values.items():
            if self._availability[attr]:
                for value in values:
                    prob = (self._heuristic_table[attr][value] * self._pheromone_table[attr][value]) / prob_accum
                    probabilities.append((prob, self._terms[attr][value]))

        return probabilities

    def size(self):
        return self._no_of_terms

    def available(self):

        if self._no_of_terms != 0:
            for attr in self._availability:
                if self._availability[attr]:
                    return True

        return False

    def sort_term(self):

        probabilities = self._get_probabilities()
        probs = [prob[0] for prob in probabilities]
        choice_idx = np.random.choice(len(probabilities), size=1, p=probs)[0]

        return probabilities[choice_idx][1]

    def update_availability(self, attr):
        self._availability[attr] = False
        return

    def get_cases(self, antecedent):

        all_cases = []
        for attr, value in antecedent.items():
            all_cases.append(self._terms[attr][value].covered_cases)

        cases = all_cases.pop()
        for case_set in all_cases:
            cases = list(set(cases) & set(case_set))

        return cases

    def pheromone_updating(self, antecedent, quality):

        # increasing pheromone of used terms
        for attr, value in antecedent.items():
            self._pheromone_table[attr][value] += self._pheromone_table[attr][value] * quality

        # Decreasing not used terms: normalization
        pheromone_normalization = self._get_pheromone_accum()
        for attr, values in self._attr_values.items():
            for value in values:
                self._pheromone_table[attr][value] = self._pheromone_table[attr][value] / pheromone_normalization

        self._reset_availability()

        return

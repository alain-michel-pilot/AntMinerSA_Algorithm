import numpy as np
from term import Term


class TermsManager:

    def __init__(self, dataset, min_case_per_rule):
        self.__terms = {}
        self.__availability = {}
        self.__attr_values = {}
        self.__pheromone_table = {}
        self.__heuristic_table = {}
        self.__no_of_terms = 0

        # build object
        self.__constructor(dataset, min_case_per_rule)

    def __constructor(self, dataset, min_case_per_rule):

        # Attrubute-Values from the entire dataset
        attr_values = dataset.attr_values.copy()
        attrs = list(attr_values.keys())

        heuristic_accum = 0

        # TABLES: { Attr : {} }
        self.__terms = {}.fromkeys(attrs, {})
        self.__pheromone_table = {}.fromkeys(attrs, {})
        self.__heuristic_table = {}.fromkeys(attrs, {})

        # TERMS: constructing __terms_dict and __availability
        self.__attr_values = {}.fromkeys(attrs, [])
        self.__availability = {}.fromkeys(attrs)
        for attr, values in attr_values.items():
            list_of_values = []
            values_dict = {}.fromkeys(values)
            for value in values:
                term_obj = Term(attr, value, dataset, min_case_per_rule)
                if term_obj.available():
                    values_dict[value] = term_obj
                    list_of_values.append(value)
                    self.__no_of_terms += 1
                    heuristic_accum += term_obj.get_heuristic()
                else:
                    values_dict.pop(value, None)

            if not list_of_values:
                self.__terms.pop(attr)
                self.__pheromone_table.pop(attr, None)
                self.__heuristic_table.pop(attr, None)
                self.__attr_values.pop(attr, None)
                self.__availability.pop(attr, None)
            else:
                self.__terms[attr] = values_dict
                self.__attr_values[attr] = list_of_values[:]
                self.__availability[attr] = True

        # No available terms on dataset
        if self.__no_of_terms == 0:
            return

        # __pheromone_table: {Attr : {Value : Pheromone}} | __heuristic_table: {Attr : {Value : Heuristic}}
        initial_pheromone = 1 / self.__no_of_terms
        for attr, values in self.__attr_values.items():
            self.__pheromone_table[attr] = {}.fromkeys(values, initial_pheromone)
            self.__heuristic_table[attr] = {}.fromkeys(values)
            for value in values:
                self.__heuristic_table[attr][value] = (self.__terms[attr][value].get_heuristic() / heuristic_accum)

        return

    def __get_prob_accum(self):

        accum = 0
        for attr, values in self.__attr_values.items():
            if self.__availability[attr]:
                for value in values:
                    accum += self.__heuristic_table[attr][value] * self.__pheromone_table[attr][value]

        return accum

    def __get_pheromone_accum(self):

        accum = 0
        for attr, values in self.__attr_values.items():
            for value in values:
                accum += self.__pheromone_table[attr][value]

        return accum

    def __reset_availability(self):

        attrs = list(self.__attr_values.keys())
        self.__availability = {}.fromkeys(attrs, True)

        return

    def __get_probabilities(self):

        prob_accum = self.__get_prob_accum()
        probabilities = []

        for attr, values in self.__attr_values.items():
            if self.__availability[attr]:
                for value in values:
                    prob = (self.__heuristic_table[attr][value] * self.__pheromone_table[attr][value]) / prob_accum
                    probabilities.append((prob, self.__terms[attr][value]))

        return probabilities

    def size(self):
        return self.__no_of_terms

    def available(self):

        if self.__no_of_terms != 0:
            for attr in self.__availability:
                if self.__availability[attr]:
                    return True

        return False

    def sort_term(self):

        probabilities = self.__get_probabilities()
        probs = [prob[0] for prob in probabilities]
        choice_idx = np.random.choice(len(probabilities), size=1, p=probs)[0]

        return probabilities[choice_idx][1]

    def update_availability(self, attr):
        self.__availability[attr] = False
        return

    def get_cases(self, antecedent):

        all_cases = []
        for attr, value in antecedent.items():
            all_cases.append(self.__terms[attr][value].covered_cases)

        cases = all_cases.pop()
        for case_set in all_cases:
            cases = list(set(cases) & set(case_set))

        return cases

    def pheromone_updating(self, antecedent, quality):

        # increasing pheromone of used terms
        for attr, value in antecedent.items():
            self.__pheromone_table[attr][value] += self.__pheromone_table[attr][value] * quality

        # Decreasing not used terms: normalization
        pheromone_normalization = self.__get_pheromone_accum()
        for attr, values in self.__attr_values.items():
            for value in values:
                self.__pheromone_table[attr][value] = self.__pheromone_table[attr][value] / pheromone_normalization

        self.__reset_availability()

        return

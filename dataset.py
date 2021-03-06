import pandas as pd
import numpy as np
from discretizer import Discretizer


class Dataset:

    def __init__(self, data, attr_survival_name, attr_event_name, attr_id_name, attr_to_ignore, discretization):
        self.survival_times = ()
        self.average_survival = None
        self.events = ()
        self.attr_values = {}
        self.data = None

        self._col_index = {}
        self._uncovered_cases = [True]*data.shape[0]
        self._original_data = data.copy()
        self._discretized_data = None
        self._discretization = discretization

        if self._discretization:
            self._discretize()

        self._constructor(attr_survival_name, attr_event_name, attr_id_name, attr_to_ignore)

    def _discretize(self):
        self._discretized_data = Discretizer().discretize(self._original_data)
        return

    def _constructor(self, attr_survival_name, attr_event_name, attr_id_name, attr_to_ignore):

        if self._discretization:
            data = self._discretized_data.copy()
        else:
            data = self._original_data.copy()

        self.survival_times = (attr_survival_name, data[attr_survival_name])
        self.average_survival = data[attr_survival_name].mean()
        self.events = (attr_event_name, data[attr_event_name])

        to_drop = [attr_survival_name, attr_event_name]
        if attr_id_name is not None:
            to_drop = to_drop + [attr_id_name]
        if attr_to_ignore:
            to_drop = to_drop + attr_to_ignore
        data.drop(columns=to_drop, inplace=True)

        col_names = list(data.columns.values)
        self.attr_values = dict.fromkeys(col_names)
        for name in col_names:
            self.attr_values[name] = list(pd.unique(data[name]))

        self._col_index = dict.fromkeys(col_names)
        for name in col_names:
            self._col_index[name] = data.columns.get_loc(name)

        self.data = np.array(data.values)

        return

    def update_covered_cases(self, covered_cases):
        # set flag for rule-covered cases
        for case in covered_cases:
            self._uncovered_cases[case] = False

        return

    def get_col_index(self, col_name):
        return self._col_index[col_name]

    def get_data(self):
        return self._original_data

    def get_data_disc(self):
        return self._discretized_data

    def get_cases_coverage(self):   # returns a bool list with True for covered cases
        return [covered == False for covered in self._uncovered_cases]

    def get_no_of_uncovered_cases(self):
        return sum(self._uncovered_cases)

    def get_instances(self):
        return list(range(len(self.data)))

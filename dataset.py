import pandas as pd
import numpy as np

from user_inputs import UserInputs


class Dataset:

    def __init__(self, data):
        self.survival_times = (UserInputs.attr_survival_name, data[UserInputs.attr_survival_name])
        self.events = (UserInputs.attr_event_name, data[UserInputs.attr_event_name])

        self.attr_values = {}
        self.data = None

        self._col_index = {}
        self._uncovered_cases = [True]*data.shape[0]
        self._original_data = data.copy()

        self.__constructor(data)

    def __constructor(self, data):

        to_drop = [UserInputs.attr_survival_name, UserInputs.attr_event_name]
        if UserInputs.attr_id_name is not None:
            to_drop = to_drop + [UserInputs.attr_id_name]
        if UserInputs.attr_to_ignore:
            to_drop = to_drop + UserInputs.attr_to_ignore
        data.drop(columns=to_drop, inplace=True)

        col_names = list(data.columns.values)
        self.attr_values = dict.fromkeys(col_names)
        for name in col_names:
            self.attr_values[name] = list(pd.unique(data[name]))

        self._col_index = dict.fromkeys(col_names)
        for idx, name in enumerate(col_names):
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

import numpy as np
from user_inputs import UserInputs
from sklearn.preprocessing import KBinsDiscretizer


class Discretizer:

    def __init__(self):
        self._method = UserInputs.discretization_method
        self._log_file = 'log_discretization.txt'
        self._save_file = 'data_discretized.csv'

    def discretize(self, data):

        if self._method == 'KBins':
            return self._k_bins(data)

    def _save_log(self, attribute, original, data_encoded, data_disc):

        f = open(self._log_file, "a+")
        f.write('Original data file: {}'.format(UserInputs.data_path))
        f.write('\nDiscretized data save file: {}'.format(self._save_file))
        f.write('\n\n ===== ATTRIBUTE DISCRETIZATION INFO =====')
        f.write('\n\n>> Attribute name: {}'.format(attribute))
        f.write('\n- Original data unique values: \n{}'.format(list(np.unique(original))))
        f.write('\n- Encoded labels: \n{}'.format(list(np.unique(data_encoded))))
        f.write('\n- Discretized labels: \n{}'.format(list(np.unique(data_disc))))
        f.close()

        return

    def _k_bins(self, original_data):    # add *kwargs variables to discretizer setting: n_bins, encode, strategy

        df_data = original_data.copy()

        # set attributes to be discretized
        if not UserInputs.attr_2disc_names:
            attrs = list(df_data.columns.values)
            attrs2remove = [UserInputs.attr_survival_name, UserInputs.attr_event_name]
            if UserInputs.attr_id_name is not None:
                attrs2remove = attrs2remove + [UserInputs.attr_id_name]
            if UserInputs.attr_to_ignore:
                attrs2remove = attrs2remove + UserInputs.attr_to_ignore
            attrs2disc = [attr for attr in attrs if attr not in attrs2remove]
        else:
            attrs2disc = UserInputs.attr_2disc_names

        # Discretization:
        enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        for attr in attrs2disc:
            to_disc = np.array(df_data[attr]).reshape(-1, 1)
            data_enc = enc.fit_transform(to_disc)
            data_disc = enc.inverse_transform(data_enc)
            if UserInputs.save_log:
                self._save_log(attr, df_data[attr], data_enc, data_disc)
            df_data[attr] = data_enc        # replaces attribute for discretized-encoded data

        if UserInputs.save_log:
            df_data.to_csv(self._save_file, index=False)

        return df_data

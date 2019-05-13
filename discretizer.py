from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np


def partial_discretizer(data_file, save_in, attr2disc, target):
    log_file = "log_disc.txt"
    f = open(log_file, "a+")
    f.write('File of discretization: ' + repr(data_file))
    f.close()

    data = pd.read_csv(data_file, delimiter=',', header=None, index_col=False)
    data = np.array(data.values)

    to_disc = data[:, attr2disc]
    targets = data[:, target]

    enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    data_enc = enc.fit_transform(to_disc, targets)
    data_disc = enc.inverse_transform(data_enc)

    for counter, value in enumerate(attr2disc):
        data[:, value] = data_enc[:, counter]

        f = open(log_file, "a+")
        f.write('\n\n= Attribute idx: ' + repr(value))
        f.write('\nEncoded labels: ' + repr(np.unique(data_enc[:, counter])))
        f.write('\nDiscretized labels: ' + repr(np.unique(data_disc[:, counter])))
        f.close()

    # cleveland_format = ['%2.1f','%d','%d','%3.1f','%3.1f','%d','%d','%3.1f','%d','%1.1f','%d','%d','%d','%d']

    np.savetxt(save_in, data, fmt='%d', delimiter=',', newline='\n') # %d = integer

    return


def complete_discretizer(data_file, save_in, target):
    log_file = "log_disc.txt"
    f = open(log_file, "a+")
    f.write('File of discretization: ' + repr(data_file))
    f.close()

    data = pd.read_csv(data_file, delimiter=',', header=None, index_col=False)
    data = np.array(data.values)

    to_disc = data[:, :-1]
    targets = np.reshape(np.array(data[:, target]), (len(data), 1))

    enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    data_enc = enc.fit_transform(to_disc, targets)
    data_disc = enc.inverse_transform(data_enc)

    for attr in range(len(data[0])-1):
        f = open(log_file, "a+")
        f.write('\n\n= Attribute idx: ' + repr(attr))
        f.write('\nEncoded labels: ' + repr(np.unique(data_enc[:, attr])))
        f.write('\nDiscretized labels: ' + repr(np.unique(data_disc[:, attr])))
        f.close()

    final_data = np.concatenate((data_enc, targets), axis=1)
    np.savetxt(save_in, final_data, fmt='%d', delimiter=',', newline='\n') # %d = integer

    return


if __name__ == '__main__':

    dataset = 'datasets/glass_preprocessed.csv'
    save_file ='datasets/glass_preprocessed_disc.csv'

    # attr_2disc = [0, 3, 4, 7, 9]
    # attr_class = 13
    # partial_discretizer(data_file=dataset, save_in=save_file, attr2disc=attr_2disc, target=attr_class)

    attr_class = 9
    complete_discretizer(data_file=dataset, save_in=save_file, target=attr_class)

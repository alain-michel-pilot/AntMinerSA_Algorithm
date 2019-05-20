
class UserInputs:   # Define necessary settings

    # DATA FILE PATH:
    header_path = 'datasets_surval/gbsg2_header.txt'
    data_path = 'datasets_surval/gbsg2.csv'

    # DATA ATTRIBUTES DESIGNATION
    attr_survival_name = 'time'
    attr_event_name = 'cens'
    attr_id_name = None
    attr_to_ignore = []

    # HEURISTIC SETTINGS:
    # Heuristic mehods: attribute_stratification,
    #                   attribute_value_stratification,
    #                   survival_average_based_entropy
    heuristic_method = 'survival_average_based_entropy'

    # KAPLAN-MEIER PARAMETERS
    kmf_alpha = 0.05        # alpha value in the confidence intervals of KM function

    # RULE PARAMETERS
    alpha = 0.05            # alpha value for statistical test confidence

    # DISCRETIZER PARAMETERS
    attr_2disc_names = ['age', 'tsize', 'pnodes', 'progrec', 'estrec']
    save_log = True
    discretization_method = 'KBins'
    # kwargs = [n_bins=, encode=, strategy=]   # insert setting variables for discretizer

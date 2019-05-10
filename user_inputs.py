
class UserInputs:   # Define necessary settings

    # DATA FILE PATH:
    header_path = 'datasets_surval/addicts_header.txt'
    data_path = 'datasets_surval/addicts.csv'

    # DATA ATTRIBUTES DESIGNATION
    attr_survival_name = 'survival_time'
    attr_event_name = 'status'
    attr_id_name = 'ID'
    attr_to_ignore = []

    # HEURISTIC SETTINGS:
    # Heuristic mehods: attribute_stratification,
    #                   attribute_value_stratification,
    #                   survival_average_based_entropy
    heuristic_method = 'survival_average_based_entropy'

# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.
    Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return l[0]
    
def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.
    Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude
    Returns:
    List of names for columns with data type specified.
    """
    return [
      tup[0]
      for tup in column_definition
      if tup[1] == data_type and tup[2] not in excluded_input_types
    ]
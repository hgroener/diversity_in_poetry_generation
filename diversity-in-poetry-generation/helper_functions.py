def flatten_list(_2d_list):
    """
    Flattens a list of lists by removing the inner list structure  

    Parameters:
    ----------
    _2d_list : Two-dimensional list

    Returns:
    -------
    flat_list : Flattened list 
    """
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

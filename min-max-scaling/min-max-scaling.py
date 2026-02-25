def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    num_rows = len(data)
    num_cols = len(data[0])
    
    scaled_matrix = [[0.0] * num_cols for _ in range(num_rows)]
    
    for col in range(num_cols):
        column_data = [data[row][col] for row in range(num_rows)]
        
        col_min = min(column_data)
        col_max = max(column_data)
        col_range = col_max - col_min
        
        for row in range(num_rows):
            if col_range == 0:
                scaled_matrix[row][col] = 0.0
            else:
                scaled_matrix[row][col] = (data[row][col] - col_min) / col_range
                
    return scaled_matrix
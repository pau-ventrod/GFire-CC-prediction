import pandas as pd
import numpy as np

## Height reading
def fix_format(df, NCOLS):
    '''
    The height document has a format error, as each row is splitted between two rows (1000 and 950 columns). 
    This function changes the format into a correct one.
    Args:
        - df: DataFrame with the wrong format ( 1 real_row = 2 df_rows )
        - NCOLS: Number of columns of the real file.
        
    Returns:
        - DataFrame with the correct format without null values.
    '''
    df = df.drop(columns=0) # Dropping first column ( Null values )
    
    even_df = df.iloc[ np.arange(0, df.shape[0], 2) ].reset_index(drop= True) # Taking even rows
    odd_df = df.iloc[ np.arange(1, df.shape[0], 2) ].reset_index(drop=True) # Taking odd rows
    
    full_df = pd.concat([even_df, odd_df], axis=1, ignore_index=True) # Concatenating both [a,b],[c,d] = [a,b,c,d]
    
    # Check all values dropped will be nan
    if full_df.iloc[:, NCOLS:].isna().sum().sum() != (full_df.shape[1] - NCOLS) * full_df.shape[0]:
        print( "Warning: dropping non-null values")
    
    full_df = full_df.drop(columns=range(NCOLS, full_df.shape[1])) # Dropping all null columns
    
    return full_df


def index_matrix(df, XCENT, YCENT, CELLSIZE):
    '''
    Fastly indexes the height matrix to change its format to x,y,z.
    Args:
        - df: Dataframe with the heights.
        - XCENT: X Coordenate of the bottom left corner.
        - YCENT: Y Coordenate of the bottom left corner.
        - CELLSIZE: Distance between two cells.
    
    Returns:
        - A dataframe with the rows in the format: x,y,z.        
    '''
    ncols = df.shape[1]
    nrows = df.shape[0]
    
    # Building a matrix for each other dimension (x and y) with the same format of z
    xdf = pd.DataFrame(np.tile(XCENT+np.arange( 0, ncols, 1)*CELLSIZE, [nrows,1]))
    ydf = pd.DataFrame(np.tile(YCENT + np.arange( nrows-1, -1, -1) * CELLSIZE, [ncols,1]).transpose())

    # Concatenating all 3 dimensions into a dataframe
    df = pd.concat([xdf.stack(), ydf.stack(), df.stack()], axis=1).reset_index(drop=True)
    df.columns = ['x_point','y_point','real_z']
    
    return df


## Manual Prediction
def CC_manual_percentage_app2(my_block, threshold, outliers, vegetation):
    '''
    De cada punt (20m x 20m), assumir que el terra d’aquella zona està a altura constant, i per tant, per a cada punt 20 x 20, 
    l’altura de cada pixel es calcularà amb: max_pixel(z)-min_punt(z). 
    %CC = %punts de vegetacio > threshold

    Args:
        - my_block: Dataframe contenin el block de dades.
        - Threshold: Threshold from which a point is considered to be from the canopy cover.
        - outliers: Number of the point-class to be considered as an outlier.
        - vegetation: Number of the point-class to be considered as vegetation.
        
    Returns:
        - Canopy Cover percentage over 100.
    '''
    if my_block.shape[0] == 0:
        return 0
    
    # Erasing outliers
    my_block = my_block[~my_block['class'].isin(outliers)]
    
    # Rounding the coordenates to correct the measurement errors
    my_block['x'] = list(map(round, my_block['x']))
    my_block['y'] = list(map(round, my_block['y']))
    
    # Agrupem cada punt per coordenades x,y, i ens quedem amb aquelles z que siguin el màxim valor
    my_block_max = my_block.groupby(['x','y'])[['z']].max().reset_index()
    my_value_min = min(my_block['z'])   
    
    my_block_max['heigh'] = my_block_max['z']-my_value_min
    
    # Adding the classes of the points
    surface_points = pd.merge(my_block_max, my_block, how='left')
    
    # Choosing vegetation points
    vegetation_points = surface_points[ surface_points['class'].isin(vegetation) ]
    
    # Canopy Cover points
    CC_points = vegetation_points[vegetation_points['heigh'] > threshold]
    
    CC_percentage = CC_points.shape[0] / my_block_max.shape[0]
    
    return CC_percentage*100

def CC_manual_percentage_app3(my_block, outliers, vegetation):
    '''
    Cada punt x,y enter té un representant.
    No usar thresholds. %CC = %punts de la classe vegetació respecte tots els altres punts representants.

    Args:
        - my_block: Dataframe contenin el block de dades.
        - outliers: Number of the point-class to be considered as an outlier.
        - vegetation: Number of the point-class to be considered as vegetation.
        
    Returns:
        - Canopy Cover percentage over 100.
    '''
    if my_block.shape[0] == 0:
        return 0
    
    # Erasing outliers
    my_block = my_block[~my_block['class'].isin(outliers)]
    
    # Rounding the coordenates to correct the measurement errors
    my_block['x'] = list(map(round, my_block['x']))
    my_block['y'] = list(map(round, my_block['y']))
    
    # Agrupem cada punt per coordenades x,y. Simplement per a llevar duplicats
    my_block_max = my_block.groupby(['x','y'])[['z']].max().reset_index()   
    
    # Adding the classes of the points
    surface_points = pd.merge(my_block_max, my_block, how='left')
    
    # Choosing vegetation points
    CC_points = surface_points[ surface_points['class'].isin(vegetation) ]
    
    CC_percentage = CC_points.shape[0] / my_block_max.shape[0]
    
    return CC_percentage*100

def CC_manual_percentage_app4(my_block, threshold, outliers, vegetation):
    '''
    Cada punt x,y enter JA NO té un representant.
    No usar thresholds. %CC = %punts de la classe vegetació respecte tots els altres punts del punt.

    Args:
        - my_block: Dataframe contenin el block de dades.
        - outliers: Number of the point-class to be considered as an outlier.
        - vegetation: Number of the point-class to be considered as vegetation.
        
    Returns:
        - Canopy Cover percentage over 100.
    '''
    if my_block.shape[0] == 0:
        return 0
    
    # Erasing outliers
    my_block = my_block[~my_block['class'].isin(outliers)]
    
    # Choosing vegetation points
    CC_points = my_block[ my_block['class'].isin(vegetation) ]
    
    CC_percentage = CC_points.shape[0] / my_block.shape[0]
    
    return CC_percentage*100
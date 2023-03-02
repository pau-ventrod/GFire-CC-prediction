import pandas as pd
import numpy as np


def get_point_coordinate_from_pixel_coordinate(pixel_coordinate, point_precision):
    """
    Adjusts the coordinates of a pixel to the coordinates of the point it corresponds to.
    
    Args:
        - pixel_coordinate: the coordinate of a pixel, only a singel axis.
        - point_precision: the precision of the groundtruth.
    Returns:
        - the coordinate of the point the pixel corresponds to.
    """
    floor_divided_pixel = pixel_coordinate // (point_precision/2)
    if floor_divided_pixel % 2:
        return floor_divided_pixel * (point_precision/2)
    else:
        return (floor_divided_pixel + 1) * (point_precision/2)
    
    
class FeatureExtractor:
    """
    Extracts the features of each point based on the pixels inside it.
    
    Attributes
    ----------
    input_data : Pandas DataFrame
        LiDAR input information with the coordinates of the points corresponding to each pixel.
    
    Methods
    -------
    mean_height(name='mean_height')
    num_pixels(name='num_pixels')
    num_ground(name='num_ground')
    num_low_vegetation(name='num_low_vegetation')
    num_medium_vegetation(name='num_medium_vegetation')
    num_high_vegetation(name='num_high_vegetation')
    num_building(name='num_building')
    num_low_point(name='num_low_point')
    max_height_diff(name='max_height_diff')
    
    """
    
    def __init__(self, input_data, block_height_grouping=6):
        
        if not set(['x_p', 'y_p', 'x', 'y', 'c', 'a']).issubset(set(input_data.columns)):
            print("Error: Input data does not have the required columns")
            
        self.xp = 'x_p'      # pixel's point x coordinate column name
        self.yp = 'y_p'      # pixel's point y coordinate column name
        self.x = 'x'         # pixel x coordinate column name
        self.y = 'y'         # pixel y coordinate column name
        self.c = 'c'         # pixel class column name
        self.a = 'a'         # pixel angle columns name

        self.data = self._get_heights(input_data, block_height_grouping)  # LiDAR data with point-assigned pixels and its height
        self.grouped = self.data.groupby([self.xp, self.yp]) # pixels grouped by point
          
        
    def _get_heights(self, input_data, grouping):
        input_data['height_merging_x'] = ( input_data[self.x] // grouping ) * grouping
        input_data['height_merging_y'] = ( input_data[self.y] // grouping ) * grouping

        input_copy = input_data.copy()

        input_copy = input_copy.groupby(['height_merging_x', 'height_merging_y'])['z'].min().reset_index()
        input_copy = input_copy.rename({'z':'surface_z'}, axis=1)

        input_data = pd.merge(input_data, input_copy, how='left', on=['height_merging_x', 'height_merging_y'])
        input_data['height'] = input_data['z'] - input_data['surface_z']

        input_data = input_data.drop(['z', 'surface_z', 'height_merging_x', 'height_merging_y'], axis=1)
        
        return input_data
        
    # Angle oriented features     
    def angle_mean(self, name='angle_mean'):
        # NEW
        '''
        Mean angle of scanning.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        '''
        values = self.grouped[self.a].mean()
        return values.rename(name, inplace=True)
    
    def angle_quantile(self, quant=0.5, name='angle_Q'):
        # NEW
        """
        Quantile value of the height array of that point.
        
        Parameters
        ----------
        quant: Percentage of the quantile. Range: [0,1]
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        values = self.grouped[self.a].quantile(quant)
        return values.rename(name+str(round(quant,2)), inplace=True)
    
    def angle_sd(self, name='angle_sd'):
        # NEW
        '''
        The standard deviation of the angle Array of that point.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        '''
        values = self.grouped[self.a].std()
        return values.rename(name, inplace=True)
    
    def angle_max_diff(self, name='angle_max_diff'):
        # NEW
        """
        The difference between the highest angle and the lowest angleu.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        values = self.grouped[self.a].max()
        return values.rename(name, inplace=True)
    
    
    # Height oriented features
    def height_quantile(self, quant, name='height_Q'):
        """
        Quantile value of the height array of that point.
        
        Parameters
        ----------
        quant: Percentage of the quantile. Range: [0,1]
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        values = self.grouped['height'].quantile(quant)
        return values.rename(name+str(round(quant,2)), inplace=True)
    
    def height_threshold_percentage(self, threshold, name='above_threshold_pct_'):
        """
        Percentage of points above a certain threshold.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        my_block = self.data.query('c in [3,4,5]').copy()
        df_canopy = my_block.query('height > @threshold').copy()

        df_canopy['counted_canopy'] = np.zeros(df_canopy.shape[0])
        df_canopy = df_canopy.groupby([self.xp, self.yp])[['counted_canopy']].count().reset_index()

        my_block['counted_no_canopy'] = np.zeros(my_block.shape[0])
        my_block = my_block.groupby([self.xp, self.yp])[['counted_no_canopy']].count().reset_index()

        my_block = pd.merge(my_block, df_canopy, how='left', on = [self.xp, self.yp])

        my_block['counted_canopy'].fillna(0, inplace=True)
        my_block[name+str(threshold)] = 100*my_block['counted_canopy'] / my_block['counted_no_canopy']
        
        my_block.drop(['counted_canopy', "counted_no_canopy"], axis=1, inplace = True)
        
        return my_block
    
    def height_sd(self, name='sd_height'):
        '''
        The standard deviation of the height Array of that point.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        '''
        values = self.grouped['height'].std()
        return values.rename(name, inplace=True)
        
    def height_max_diff(self, name='max_height_diff'):
        """
        The difference between the highest LiDAR point and the lowest LiDAR point.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        values = self.grouped['height'].max()
        return values.rename(name, inplace=True)
    
    def height_second_max(self, name='height_second_max'):
        """
        The second highest LiDAR point.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        values = self.grouped['height'].nlargest(2).reset_index().drop('level_2', axis=1)
        
        return values.rename({'height':name}, axis=1)
    
    def height_mean(self, name='mean_height'):
        """
        Mean height with respect to the geoid.
         
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        values = self.grouped['height'].mean()
        return values.rename(name, inplace=True)
    
    # Points (pixels) oriented features
    def pts_number(self, name='num_pixels'):
        """
        Number of pixels of each point.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        values = self.grouped.count()[self.x]
        return values.rename(name, inplace=True)
    
    def pts_num_ground(self, name='num_ground'):
        """
        Number of pixels classified as "ground".
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        filtered = self.data[self.data[self.c]==2] # 2 corresponds to "low vegetation"
        values = filtered.groupby([self.xp, self.yp]).count()[self.x]
        return values.rename(name, inplace=True)
    
    def pts_num_low_vegetation(self, name='num_low_vegetation'):
        """
        Number of pixels classified as "low vegetation".
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        filtered = self.data[self.data[self.c]==3] # 3 corresponds to "low vegetation"
        values = filtered.groupby([self.xp, self.yp]).count()[self.x]
        return values.rename(name, inplace=True)
    
    def pts_num_medium_vegetation(self, name='num_medium_vegetation'):
        """
        Number of pixels classified as "medium vegetation".
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        filtered = self.data[self.data[self.c]==4] # 4 corresponds to "low vegetation"
        values = filtered.groupby([self.xp, self.yp]).count()[self.x]
        return values.rename(name, inplace=True)
    
    def pts_num_high_vegetation(self, name='num_high_vegetation'):
        """
        Number of pixels classified as "high vegetation".
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        filtered = self.data[self.data[self.c]==5] # 5 corresponds to "high vegetation"
        values = filtered.groupby([self.xp, self.yp]).count()[self.x]
        return values.rename(name, inplace=True)
    
    def pts_num_building(self, name='num_building'):
        """
        Number of pixels classified as "building".
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        filtered = self.data[self.data[self.c]==6] # 6 corresponds to "building"
        values = filtered.groupby([self.xp, self.yp]).count()[self.x]
        return values.rename(name, inplace=True)
    
    def pts_num_low_point(self, name='num_low_point'):
        """
        Number of pixels classified as "low point".
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)
        """
        filtered = self.data[self.data[self.c]==7] # 7 corresponds to "low point"
        values = filtered.groupby([self.xp, self.yp]).count()[self.x]
        return values.rename(name, inplace=True)
    
    def pts_not_cassified(self, name='not_classified_pts'):
        '''
        Number of points that are not vegetation nor ground.
        
        Parameters
        ----------
        name : name of the resulting series object (that will eventually be the column name in the dataset)        
        '''
        aux_arr = self.data.query('c not in [3, 4, 5, 6, 7]').copy()
        
        aux_arr = aux_arr.groupby([self.xp, self.yp])[self.c].count()
        
        return aux_arr.rename(name)

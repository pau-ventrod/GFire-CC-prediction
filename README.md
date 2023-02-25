# GFire-CC-prediction

This is part of a bigger project called GFire (https://github.com/GFire-UAB). The main goal of GFire is to predict how a forest fire will spread throughout time in order to help firefighters stop the fire.

This model needs to be fed with some parameters, such as humidity of the zone, canopy cover, fuel types, ... 
Our work consists in building Machine Learning models to predict some of the parameters using LiDAR (high resolution laser) data. 

We are currently working on Canopy Cover prediction. Canopy Cover is described as the percentage of the area that is covered with canopy as seen from the sky.

### Datasets information

Using data from the ICGC (Institut Cartogràfic i Geològic de Catalunya) and the open-source geographic information system QGIS, we have developed (as per now) a total of 3 different datasets that we are using to develop and train our models. The data corresponds to different regions of Catalonia (Spanish autonomous community).

All the datasets consist of an **input** `.txt` file corresponding to a LiDAR point cloud, and a **groundtruth** `.csv` file, which contains the canopy cover percentage for a given region of 20x20 meters. The minimum point density of the ICGC LiDAR point cloud is 0.5 points per meter squared.

- **Tiny dataset**: 100x100 meters. Mainly for debugging.
  - Input attributes: for each LiDAR data point, the x,y,z coordinates and its classification.
  - Groundtruth attributes: for every region of 20x20 meters, the x,y coordinates of the center of the region and the percentage of canopy cover (CC).
  
- **Toy dataset**: 680x560 meters. A "tipical" region of Catalonia. 
  - Input attributes: for each LiDAR data point, the x,y,z coordinates and its classification.
  - Groundtruth attributes: for every region of 20x20 meters, the x,y coordinates of the center of the region and the percentage of canopy cover (CC).
  
- **Nonzerogt dataset**: 600x600 meters. The groundtruth only contains positive values.
  - Input attributes: for each LiDAR data point, the x,y,z coordinates and its classification.
  - Groundtruth attributes: for every region of 20x20 meters, the x,y coordinates of the center of the region and the percentage of canopy cover (CC).

### Current scores
The Score metric used will be r-squared, as it gives a value between (-1, 1), being 1 the best possible value, that shows how correlated are two arrays (in this case the predicted and the real values of the canopy cover).


|                   | Toy Dataset | Nonzerogt Dataset |
|-------------------|-------------|-------------------|
| Threshold method  | 0.754       | 0.485             |
| Vegetation method | 0.675       | 0.392             |


Note that the Nonzerogt Dataset has less accuracy than the others. This is due to the fact that having 0 values on the groundtruth apparently drastically increases the metric value.


### Members of the team:

- Alejandro Donaire: aledonairesa@gmail.com | https://www.linkedin.com/in/alejandro-donaire

- Èric Sánchez: ericsanlopez@gmail.com | 

- Pau Ventura: pau.ventura.rodriguez@gmail.com | https://www.linkedin.com/in/pauvr

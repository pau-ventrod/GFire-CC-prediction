# GFire-CC-prediction

This is part of a bigger project called GFire (https://github.com/GFire-UAB). The main goal of GFire is to predict how a forest fire will spread throughout time in order to help firefighters stop the fire.

This model needs to be fed with some parameters, such as humidity of the zone, canopy cover, fuel types, ... 
Our work consists on building Machine Learning models to predict some of the parameters using LiDAR (high resolution laser) data. 

We are currently working on Canopy Cover prediction. Canopy Cover is described as the percentage of the area that is covered with canopy if it is seen from the sky.

### Datasets information

(ALE)


### Current scores
The Score metric used will be r-squared, as it gives a value between (-1, 1), being 1 the best possible value, that shows how correlated are two arrays (in this case the predicted and the real values of the canopy cover).


|                   | Toy Dataset | Nonzerogt Dataset |
|-------------------|-------------|-------------------|
| Threshold method  | 0.754       | 0.485             |
| Vegetation method | 0.675       | 0.392             |


Note that the Nonzerogt Dataset has less accuracy than the others. This is due to the fact that having 0 values on the groundtruth apparently drastically increases the metric value.


### Members of the team:

- Alejandro Donaire:

- Èric Sánchez: ericsanlopez@gmail.com | 

- Pau Ventura: pau.ventura.rodriguez@gmail.com | https://www.linkedin.com/in/pauvr

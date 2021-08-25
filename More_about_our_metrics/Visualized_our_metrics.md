Optimize the comparison between a single example and Chamfer-Distance.
- We can see that Chamfer-distance does not decrease monotonically during our optimization process.
- The optimal result of our optimization does not correspond to the smallest Chamfer distance.

The above two points show that our metric is essentially different from Chamfer distance optimization.

For some examples, it may be better to use Chamfer distance in combination with our metric.
As shown in the two examples in the following figure, we can combine the two to speed up the convergence and sometimes avoid the generation of local optimal solutions.

As 

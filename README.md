# graphite_swe
Files for the graphite digital technical focusing on the SWE track (ride share optimization algorithm)

The Python file has a class called "RidePricingOptimizer" that holds all the methods related to the optimization algorithm. 

The algorithm is a finite horizon dynamic programming algorithm that determines dynamic pricing tables for each time step
based on input parameters.

To run the optimization algorithm directly within the script, adjust class initialization values in main(): T, price_options, 
per_ride_op_cost, spillover_fraction, backlog_step, backlog_max, end_backlog_penalty, base_demands, and drivers.

The density pricing python script is a small extension that allows for a surcharge to be added to rides beginning or ending in sparsely populated areas, which reflects the increased overhead costs for opperating in these areas. This script is not integrated with the algorithm; it simply provides functions that return True or False. The script could be integrated with a more comprehensive pricing platform to add the surcharge. There are two .csv files that provide the necessary data for the script. 

NOTE: All files related to density pricing were uploaded a few minutes after the deadline (6:10pm). ALL files that fulfilled necessary requirements (algorithm file and original readme) were uploaded and submitted on the form before the 6pm deaadline. I wanted to add this extra extension for fun before, but I unfortunately had to go to class before I could finish it all up. It is supplementary to the main assignment. 

Direct any questions to mgerber@g.hmc.edu

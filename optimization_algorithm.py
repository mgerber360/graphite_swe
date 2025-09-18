# imports
import numpy as np
import pprint

"""
A class representing the optimization algorithm for a specific set of parameters.

Instance variables: T (int), price_options (np.ndarray), per_ride_op_cost (float), spillover_fraction (float), backlog_step (float), backlog_max (float), end_backlog_penalty (float), base_demands (List[float]), drivers (List[int])
Methods: _make_backlog_arr, _snap_to_grid, _grid_index, find_dp, simulate_day, _calculate_demand_and_spillover
"""
class RidePricingOptimizer:

    def __init__(self, T, price_options, per_ride_op_cost, spillover_fraction, backlog_step, backlog_max, end_backlog_penalty, base_demands, drivers):
        self.T = int(T)  # number of time periods
        self.price_options = price_options.astype(float)  # list of possible prices
        self.per_ride_op_cost = float(per_ride_op_cost)  # operational cost per ride
        self.spillover_fraction = float(spillover_fraction) # fraction of unserved demand that spills over (set to 1 normally)
        self.backlog_step = float(backlog_step)  # step size for backlog grid
        self.end_backlog_penalty = float(end_backlog_penalty)  # penalty for having a backlog at the end of the day
        self.backlog_max = float(backlog_max) # maximum backlog possible
        self.base_demands = base_demands # baseline demand for each period
        self.drivers = drivers # number of drivers available

        # validate inputs
        assert isinstance(self.T, int) and self.T > 0
        assert self.price_options.size > 0 and np.isfinite(self.price_options).all()
        assert len(self.base_demands) == self.T
        assert len(self.drivers) == self.T and np.all(np.asarray(self.drivers) >= 0)
        assert self.backlog_step > 0 and self.backlog_max >= self.backlog_step
        assert 0.0 <= self.spillover_fraction <= 1.0 and self.end_backlog_penalty >= 0

    def _make_backlog_arr(self):
        """
        Build a discretized 1D array to store backlog level
        
        Args: None
        Returns: backlog_arr (np.ndarray)
        """
        n = int(self.backlog_max // self.backlog_step) # find the number of elements that should be in the array
        arr = np.round(self.backlog_step * np.arange(n + 1), 6) # build array
        return arr
    
    def _snap_to_grid(self, x):
        """
        Snap a value x to the nearest point in the backlog array
        
        Args: x (float)
        Returns: snapped_value (float)
        """
        clipped = np.clip(x, 0, self.backlog_max) # ensure x is within bounds
        snapped_value = np.round((np.rint(clipped / self.backlog_step) * self.backlog_step), 6) # find the value in the backlog array that x is closest to
        return snapped_value
    
    def _grid_index(self, backlog_value):
        """
        Convert a snapped backlog value to its index in the backlog array

        Args: backlog_value (float)
        Returns: index (int)
        """
        if backlog_value in self.index_dict: # look up in dictionary to quickly (O(1)) find the index
            return self.index_dict[backlog_value]
        b_resnapped = self._snap_to_grid(backlog_value) # avoid floating point issues
        return self.index_dict[b_resnapped]
    
    def find_dp(self):
        """
        Builds and solves a profit dynamic programming (DP) table representing the profit at every time and backlog state that has shape (T+2, num_backlog_states) where P[T+1] is the final period.
        Also builds a DP table representing the price decsision at every time and backlog state with shape (T+1, num_backlog_states).

        Args: None
        Returns: (profit_table, price_table) (tuple)
        """
        # construct a discretized backlog array
        self.backlog_arr = self._make_backlog_arr()

        num_backlog_states = self.backlog_arr.shape[0]
        num_price_options = self.price_options.shape[0] 

        # create a dict for _grid_index method
        self.index_dict = {backlog_value: indx for indx, backlog_value in enumerate(self.backlog_arr)}

        # initialize our profit table
        profit_table = np.full((self.T + 2, num_backlog_states), -np.inf, dtype=float)
        # initialize our price table
        price_table = np.zeros((self.T + 1, num_backlog_states), dtype=int)

        # penalize leftover backlog at the end of the day so that the pricing model tries to clear it
        profit_table[self.T + 1, :] = -self.end_backlog_penalty * self.backlog_arr

        # backwards induction over all the time periods
        for t in range(self.T, 0, -1):
            # check for each possible backlog value at each time period
            for backlog_indx, backlog in enumerate(self.backlog_arr):
                best_profit = -np.inf
                best_price_indx = 0
                # try every price possibility and choose the one that maximizes now + future profit (Bellman Equation)
                for price_indx, price in enumerate(self.price_options):
                    fresh_demand = self.base_demands[t - 1] # get the new demand for the timer period
                    total_riders = fresh_demand + backlog # total riders is new demand + backlog
                    served, unserved = self._calculate_demand_and_spillover(total_riders, price, t-1) # calculate the number of unserved customers because of price concerns and served customers
                    spillover = self.spillover_fraction * unserved # calculate the spillover
                    future_backlog = self._snap_to_grid(spillover) # find the closest backlog array value

                    immediate_profit = (price - self.per_ride_op_cost) * served # calculate the immediate profit
                    future_profit = profit_table[t + 1, self._grid_index(future_backlog)] # look up the future profit given the future backlog 
                    total_profit = immediate_profit + future_profit # total profit is immediate + future according to Bellman equation

                    # keep the best price
                    if total_profit > best_profit:
                        best_profit = total_profit
                        best_price_indx = price_indx

                # store the best achievable profit and the corresponding price in the tables
                profit_table[t, backlog_indx] = best_profit
                price_table[t, backlog_indx] = best_price_indx

        return profit_table, price_table
        
    def simulate_day(self, price_table, initial_backlog=0.0):
        """
        Uses the dynamic programming tables found in the find_dp method to determine the optimal prices over the entire time span given
        Args: price_table (np.ndarray), initial_backlog (float) = 0
        Returns: history (Dict[str, List[float]])
        """
        backlog = self._snap_to_grid(initial_backlog) # snap initial backlog to backlog array

        history = {
            'prices': [],
            'demands': [],
            'served': [],
            'unserved': [],
            'backlogs': [],
            'profits': []
        }

        # go through all times with the precomputed price table
        for t in range(1, self.T + 1):
            # find the best price for this time and backlog
            price_indx = price_table[t, self._grid_index(backlog)]
            optimal_price = float(self.price_options[price_indx])

            fresh_demand = self.base_demands[t - 1] # get the new demand for the time period
            total_riders = fresh_demand + backlog # total riders is new demand + backlog
            unserved = self._calculate_demand_and_spillover(total_riders, optimal_price, t-1)[1] # calculate the number of unserved customers because of price concerns
            served = total_riders - unserved # calculate the number of served customers
            spillover = self.spillover_fraction * unserved # calculate the spillover

            profit = (optimal_price - self.per_ride_op_cost) * served # calculate the optimized profit from this time period

            # save what happened in this period
            history['prices'].append(optimal_price)
            history['demands'].append(total_riders)
            history['served'].append(served)
            history['unserved'].append(unserved)
            history['backlogs'].append(backlog)
            history['profits'].append(profit)

            backlog = self._snap_to_grid(spillover) # find the closest backlog array value

        # acount for backlog penalty at the end
        total_profit = sum(history['profits']) 
        history['total_profit'] = [total_profit]

        return history
    
    def _calculate_demand_and_spillover(self, total_demand, price, t):
        """
        Calculate actual ridership and spillover based on price.
        - Low prices (<=$10): Everyone rides, no deferrals
        - Medium prices ($10<p<=$20): 10% of customers defer to next period  
        - High prices ($20+): 30% of customers defer to next period
        
        Args: total_demand (int), price (int)
        Returns: (actual_customers_this_period, customers_deferred_to_next_period)
        """
        if price <= 10:
            if total_demand > self.drivers[t]:
                return self.drivers[t], total_demand - self.drivers[t]
            return total_demand, 0
        elif price <= 20:
            deferred = round(total_demand * 0.1)
            actual = total_demand - deferred
            if actual > self.drivers[t]:
                actual = self.drivers[t]
                deferred = total_demand - actual
            return actual, deferred
        else:  # price > 20:
            deferred = round(total_demand * 0.3)
            actual = total_demand - deferred
            if actual > self.drivers[t]:
                actual = self.drivers[t]
                deferred = total_demand - actual
            return actual, deferred



# main function for testing
def main():

    # makes everything print nice
    def to_python_scalars(obj):
        if isinstance(obj, dict):
            return {k: to_python_scalars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python_scalars(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    # create optimization function
    optimizer = RidePricingOptimizer(
        T=2,
        price_options = np.array([8,12, 22]),
        per_ride_op_cost=0,
        spillover_fraction=1.0,
        backlog_step=1,
        backlog_max=5,
        end_backlog_penalty=30,
        base_demands=[10,10],
        drivers = np.array([8,8])
    )
    
    print(f"Base demands: {optimizer.base_demands}")
    print(f"Available prices: {optimizer.price_options}")
    print(f"Cost per ride: ${optimizer.per_ride_op_cost}\n")
    
    # Test simulation
    profit_table, price_table = optimizer.find_dp()
    results = optimizer.simulate_day(price_table)
    clean = to_python_scalars(results)
    pprint.pprint(clean)

# Sample usage and testing
if __name__ == "__main__":
    main()
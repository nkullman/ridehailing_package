# TODOs and User Notes

## TODOs

### Shorter-term, higher priority

- Fix rendering
- Improve documentation
- Query the NYC taxi records on OpenData to determine, for each origin_zone-destination_zone pair and each hour of the day (our current time-discretization for travel times), the mean and standard deviation of the trip times between those zones. Use this to get a more realistic `speeds` dataset.

### Longer term, lower priority

- Have `render` take an `obs=None` argument that, if not `None`, would call a special `_render_obs` function that would plot the state `obs` rather than the environment's current state. (Is this feasible?)
- Investigate ways to sample faster for getting request indices and then the subsequent `.loc` call. These seem to be what takes awhile in `reset` (will these be faster on subsequent `reset` calls? just slow for the first episode?)
- Allow user to pass in filenames for requests, lots, geometries, environment's parameters, etc.
- Make our action and observation spaces subclass `gym.Spaces.Dict`, and override their .sample() function
- Set `_EPS_START_HR` according to the trips file loaded: use the value of the hour that marks the beginning of the slowest 2-hr window
- Generalize discretized time from hour to any arbitrary time unit

## User Notes

### Actions

- To give a vehicle a null repositioning assignment, set its repositioning value equal to `env.num_lots`
- To give a request a null vehicle assignment (to wait to assign it to a vehicle), set its assigning vehicle to `self.num_vehicles`
- Request rejections take precedence over request assignments (if a request is rejected, any assignment of a vehicle to that request is ignored)

# TODOs

## Shorter-term, higher priority

- Update setup.py to require `Pillow>=8.0.0` and `matplotlib`
- Improve documentation

## Longer term, lower priority

- Have `render` take an `obs=None` argument that, if not `None`, would call a special `_render_obs` function that would plot the state `obs` rather than the environment's current state. (Is this feasible?)
- Allow user to pass in filenames for lots, geometries, environment's parameters, etc.
- Make our action and observation spaces subclass `gym.Spaces.Dict`, and override their .sample() function (and tie this function to the environment's seeding)
- Set `_EPS_START_HR` according to the trips file loaded: use the value of the hour that marks the beginning of the slowest 2-hr window
- Generalize discretized time from hour to any arbitrary time unit
- Make the action space simpler: have a single request element, with a special value indicating "not providing assignment yet" (`self._V`?) and another special value indicating "reject this request" (`self._V+1`?).

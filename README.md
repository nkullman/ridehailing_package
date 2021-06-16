# pyhailing

An OpenAI gym environment for the control of a ridehailing fleet.

*This environment supports the [DIMACS Challenge's](http://dimacs.rutgers.edu/programs/challenge/vrp/) dynamic VRP variant. More information coming soon.*

## Introduction

*pyhailing* provides a simple ridehailing environment in which an agent controls a homogeneous fleet of vehicles that serve randomly arising trip requests. An agent's goal is to maximize profit over a 24 hr period, calculated as the revenue earned from the service of requests, minus travel costs.

The environment is based on data from business days in 2018 on the island of Manhattan in New York City.

## Installation and Usage

pyhailing can be installed via `pip install pyhailing`, and an environment can be initiated as follows:

```python
from pyhailing import RidehailEnv

env = RidehailEnv()
env.reset()
...
```

It can then be used like other [OpenAI Gym](https://gym.openai.com/) environments.

## Additional Details

There are some important nuances to be aware of with pyhailing. For starters, the action and observation spaces are dynamic. More information on this and more follows in this section.

### Observation Space

Agents' receive observations of the environment in the form of a Python dictionary with the following keys:

- `time`: The current time, a float between 0 and 86400 (24 hours in seconds).
- `dow`: The day of the week, an integer (starting at 0 for Monday).
- `request_locs`: The locations of pending requests. This is a 3D float array with shape `(num_pending_requests, 2, 2)`.
  1. The first dimension corresponds to requests that are available for assignment to a vehicle in the fleet. As the number of assignable requests can vary by step, **the size of this dimension is dynamic**.
  1. The second dimension indexes requests' locations: its origin (0) and destination (1).
  1. The third dimension indexes the coordinates of a request location: x (0) and y (1).
- `request_times`: The times at which requests were released (when the agent first became aware of them). This is a 1D float array with shape `(num_pending_requests,)`.
- `v_locs`: Vehicles' locations. This is a 2D float array with shape `(num_vehicles, 2)`.
  1. The first dimension indexes vehicles in the fleet.
  1. The second dimension indexes the coordinates of their locations: x (0) and y (1).
- `v_jobs`: Vehicles' jobs' types. Vehicles maintain a queue of up to three jobs, where the first job is the one that they are currently performing. These jobs are described by an integer where **0 represents idling** (when a vehicle is sitting at a designated idling location, usually referred to as a "lot"); **1 represents repositioning** (when a vehicle is en route to a lot at which they will begin to idle); **2 represents setup** (when a vehicle is on their way to pick up a customer); **3 represents processing** (when a vehicle is transporting a customer to their destination); and **4 represents the null job** (when a vehicle lacks any job). `v_jobs` is a 2D integer array with shape `(num_vehicles, 3)`.
  1. The first dimension indexes vehicles in the fleet.
  1. The second dimension indexes vehicles' jobs: first job (0), second job (1), and third job (2).

- `v_job_locs`: The locations for vehicles' jobs. This is 4D float array with shape `(num_vehicles, 3, 2, 2)`.
  1. The first dimension indexes vehicles in the fleet.
  1. The second dimension indexes vehicles' jobs: first job (0), second job (1), and third job (2).
  1. The third dimension indexes jobs' locations: their origins (0) and destinations (1).
  1. The fourth dimension indexes locations' coordinates: x (0) and y (1).

### Action Space

Agents should specify actions as a dictionary... **TODO AM HERE**
```python

```

Additional details:

- To give a vehicle a null repositioning assignment, set its repositioning value equal to `env.num_lots`
- To give a request a null vehicle assignment (to wait to assign it to a vehicle), set its assigning vehicle to `self.num_vehicles`
- Request rejections take precedence over request assignments (if a request is rejected, any assignment of a vehicle to that request is ignored)

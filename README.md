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

Agents can then interact with the environment as with any other [OpenAI Gym](https://gym.openai.com/) environment.

## Usage Details

At a high-level, the agent receives observations of the environment that include information about what time it is, where any pending requests are and how long they've been waiting, where its vehicles are, and what its vehicles are doing.

The agent then chooses actions that specify which of the pending requests they are going to reject, which vehicles they want to assign to the pending requests, and to which lots vehicles should be repositioned to begin idling while they wait for future requests.

### Observation Space

Agents' receive observations of the environment as a dictionary with the following keys:

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
- `v_jobs`: Vehicles' jobs' types. Vehicles maintain a queue of up to three jobs, where the first job is the one that they are currently performing. These jobs are described by an integer where **0 represents idling** (when a vehicle is sitting at a designated idling location, usually referred to as a "lot"); **1 represents repositioning** (when a vehicle is en route to a lot at which they will begin to idle); **2 represents setup** (when a vehicle is on their way to pick up a customer); **3 represents processing** (when a vehicle is transporting a customer to their destination); and **4 represents the null job** (when a vehicle lacks a specified job). `v_jobs` is a 2D integer array with shape `(num_vehicles, 3)`.
  1. The first dimension indexes vehicles in the fleet.
  1. The second dimension indexes vehicles' jobs: first job (0), second job (1), and third job (2).

- `v_job_locs`: The locations for vehicles' jobs. This is 4D float array with shape `(num_vehicles, 3, 2, 2)`.
  1. The first dimension indexes vehicles in the fleet.
  1. The second dimension indexes vehicles' jobs: first job (0), second job (1), and third job (2).
  1. The third dimension indexes jobs' locations: their origins (0) and destinations (1).
  1. The fourth dimension indexes locations' coordinates: x (0) and y (1).

### Action Space

Agents specify actions as a dictionary with the following keys:

- `req_rejections`: Whether to reject pending requests. This is a 1D binary integer array with shape `(num_pending_requests, )`. Zeros indicate that the agent does not want to reject a pending request; ones that they do. Because the number of pending requests can vary by step, **the size of this array is dynamic**.
- `req_assgts`: Assignments of vehicles to pending requests. This is a 1D integer array with shape `(num_pending_requests, )`. A value of `v` at index `i` in this array specifies that the `v`-th vehicle should serve the `i`-th pending request. Valid values for assignment are in `{0..num_vehicles-1}`; if the agent wishes to wait to assign a vehicle to the request ("I might want to assign a vehicle to this request, but let's wait and see if any other requests pop up soon"), then they indicate this by providing an "assignment" value equal to `num_vehicles`. Because the number of pending requests can vary by step, **the size of this array is dynamic**.
- `reposition`: Where to reposition the vehicles. Vehicles may be assigned to any of the designated idling locations ("lots"). (The locations of these lots are available in `env.lots`.) A value of `l` at index `i` in this array specifies that the `i`-th vehicle should begin repositioning towards lot `l`; once it arrives to that location, it begins idling until it receives another instruction. If the agent does not wish to reposition the `i`-th vehicle, then the `i`-th entry in the array should be equal to the number of lots (`env.num_lots`).

### Additional Information and Caveats

- When a vehicle is given a request assignment, it immediately begins traveling to the customer, and then immediately begins serving the customer. When it is done, it triggers a new decision epoch.
- The exception to the above rule is if a vehicle is already serving a customer, in which case it departs for the new customer as soon as it drops off its current customer.
- A vehicle that is en route to pickup a customer (has a first job of type 2=setup) cannot be assigned a subsequent request. However, a vehicle that is currently serving a customer (has a first job of type 3=processing) is eligible to be assigned to a subsequent request.
- Customers must be picked up within 5 minutes of when they first request. If a vehicle is assigned to a request and their expected arrival time to that request is more than 5 minutes, then the assignment is ignored.
- Agents' specified request rejections take precedence over request assignments (if a request is marked as rejected, then any assignment of a vehicle to that request is ignored).
- Repositioning instructions are ignored for vehicles that are serving requests.
- Assignments of vehicles to requests are permanent -- it is not possible to cancel an assignment or re-assign it to another vehicle.
- If a vehicle runs out of assigned jobs (when its first job type is 4=null), it MUST receive a job assignment in the next decision epoch.
- By default, decision epochs ("steps") in the environment are only triggered by two types of events: 1) the arrival of a new request, and 2) when a vehicle completes the service of a request and has no subsequently scheduled jobs. The `max_interdecision_time` argument changes this behavior, triggering a new decision epoch if more than `max_interdecision_time` seconds have elapsed.
- By default, agents are allowed unlimited wall-clock time to provide an action from a given observation. This is not especially realistic. For those looking for more of a challenge, try specifying a value for the `action_timelimit` argument. If the agent takes more than `action_timelimit` seconds (of wall-clock time) to specify an action, then the episode terminates with a reward of negative infinity.
- Agents can sample random actions from the environment via `env.get_random_action()`. Sampling directly from the action space does not work (i.e., do not use the typical `env.action_space.sample()`)

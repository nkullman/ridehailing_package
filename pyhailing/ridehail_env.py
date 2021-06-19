from typing import List, Optional, Tuple, Union
import datetime
import logging
from numpy import random
import pkg_resources
import time

import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyhailing.core import RidehailGeometry, Jobs


class RidehailEnv(gym.Env):
    """
      Defines the Ridehail environment.

      The environment defines which actions can be taken at which point and
      when the agent receives which reward.

      In the Ridehail environment, there are random requests for rides from
      some origin location to some destination location. An agent in this
      environment controls a (homogeneous) fleet of vehicles that service
      these requests. In addition to serving requests, the vehicles can be moved
      around to wait at designated locations ("lots") in anticipation of future
      requests.

    """

    _EPS_START_HR = 3
    _EPS_START_T_WINDOW = 8
    _NUM_T_WINDOW = 24 * 4
    _HRS_PER_DAY = 24
    _S_PER_MIN = 60
    _S_PER_HR = 3600
    _YR = 2018  # The year of the trip data.

    _MAX_TIME = 86400 # one day in seconds
    _NEVER = _MAX_TIME + 1
    _MAX_WAIT = 300 # 5 min in seconds
    _TRAVEL_COST = 0.53 # $/km
    _FIXED_REWARD = 10.75 # $/km
    _VARIABLE_REWARD_DIST = 4.02 # $/km

    _TIME_STR_FORMAT_PRINT = '%H:%M:%S (%a %b %d)'


    def __init__(self,
        num_vehicles: int=20,
        num_requests: int=1000,
        stochastic: bool=False,
        distances: str="manhattan",
        seed: int=321,
        action_timelimit: float=np.inf,
        max_interdecision_time: Optional[float]=None,
    ):
        """Instantiates a ridehailing environment.
        
        Inputs:
        
            num_vehicles: The number of vehicles in the agent's fleet.
            
            num_requests: The number of requests that should arise over a 24 hr period.
            
            stochastic: Whether vehicles' travel times should be subject to randomness
                (NOTE: stochastic travel times are not yet implemented)

            distances: How to measure distances between locations. Options are
                "euclidean" and "manhattan"

            seed: A seed for the environment's uncertainties.
                (NOTE: This applies to environment dynamics but does not yet apply
                to the sampling of random states/actions from the environment's
                state/action spaces.)

            action_timelimit: How much wall-clock time (in seconds) the agent is
                permitted to provide an action. Default (inf) is no time limit.

            max_interdecision_time: The maximum amount of (simulated) time to allow
                between decision epochs (in seconds). The default behavior is to
                only trigger decision epochs upon the arrival or completion of a request.
                
        """

        logging.info("Ridehail environment being initialized...")

        self._num_vehicles = self._V = num_vehicles
        self._num_requests = self._R = num_requests
        self._stochastic = stochastic
        self._distances = distances
        self._seed = seed
        self._action_timelimit = action_timelimit
        self._max_interdecision_time = max_interdecision_time if max_interdecision_time is not None else self._NEVER

        self._load_trips()
        
        self._geom = RidehailGeometry(seed, distances, stochastic)
        self._geom.add_init_weights_to_lots(self._trips_df, self._EPS_START_T_WINDOW)


        # What episode are we on
        self.curr_episode = -1
        # What step is the agent on in the current episode
        self.curr_step = -1
        # The time at which the last observation was released to the agent.
        self._obs_release_time = None
        # Note that there are no pending requests (used in the initialization of
        # our action and observation spaces).
        self.num_pending_requests = 0
        # Initialize our observation and action spaces
        self._make_observation_space()
        self._make_action_space()

        # Initialize rendering
        self._prep_rendering()

        # Perform whatever initial seeding needs to be done.
        self._initial_seeding()


    def _initial_seeding(self) -> None:

        # We're going to have 2 distinct generators:
        # One for initializing vehicle locations at the beginning of episodes.
        # And one for sampling requests.

        # Create a seed spawner. This will be called twice for each episode:
        # To create a new seed for our vehicle-placement generator, and to
        # create a new seed for our request-sampling generator.

        self._seed_spawner = np.random.SeedSequence(self._seed)

        # Initialize our generators to None. They'll be properly initialized
        # when the environment is reset.
        self._request_sampler = None
        self._vehicle_sampler = None


    def _reseed(self) -> None:
        """Reseeds all RNGs with a new seed."""

        # Use our generator to create two new seeds.
        spawns = self._seed_spawner.spawn(2)

        # Use them to seed our vehicle and request sampler for the episode.
        self._vehicle_sampler = np.random.default_rng(spawns[0])
        self._request_sampler = np.random.default_rng(spawns[1])

        # Perform the seeding for our geometery as well.
        self._geom.reseed()

    
    def _get_pct_trips_by(self, cols: List[str]) -> pd.DataFrame:
        """Get the percentage of trips by the last dimension in cols."""

        trips_by_cols = self._trips_df.groupby(cols)["n_trips"].sum()

        if len(cols) == 1:
            return trips_by_cols / trips_by_cols.sum()
        else:
            return trips_by_cols / self._trips_df.groupby(cols[:-1])["n_trips"].sum()

    
    def _load_trips(self) -> None:
        """Load the CSV containing trips data and save some useful info for later."""

        trips_fname = pkg_resources.resource_stream(__name__, 'data/trips.csv').name
        self._trips_df = pd.read_csv(trips_fname, dtype=int)

        # Record the percentage of trips by weekday
        self._dow_wts = self._get_pct_trips_by(["dow"])


    @property
    def x_range(self):
        return self._geom.x_range


    @property
    def y_range(self):
        return self._geom.y_range
    

    @property
    def _NULL_X(self):
        return self.x_range[0]
    

    @property
    def _NULL_Y(self):
        return self.y_range[0]


    @property
    def num_lots(self):
        return self._geom.num_lots


    @property
    def _D(self):
        """A convenience getter for num_lots."""
        return self.num_lots


    @property
    def num_zones(self):
        return self._geom.num_zones


    @property
    def _Z(self):
        """A convenience getter for "num_zones"."""
        return self.num_zones

    
    @property
    def _lots(self):
        return self._geom.lots


    @property
    def lots(self):
        return self._geom.lots.loc[:, ["x", "y"]].copy()

    
    @property
    def _zones(self):
        return self._geom.zones
    
    
    @property
    def curr_state(self):
        """Returns the current state of the environment."""
        return self._make_state()

    @property
    def num_vehicles(self):
        """Number of vehicles in the fleet."""
        return self._num_vehicles
    
    
    def _pending_requests_mask(self) -> pd.Series:
        """Provides the boolean mask for requests that are available for assignment."""
        return (
                self._requests["released"]
                & ~self._requests["rejected"]
                & self._requests["vehicle"].isna()
            )
    
    
    def _get_pending_requests(self) -> pd.DataFrame:
        """Returns all active requests."""
        return self._requests.loc[self._pending_requests_mask(), :]

    
    def _get_num_pending_requests(self) -> int:
        """Returns the number of requests currently pending."""

        return self._pending_requests_mask().sum()

    
    def _make_action_space(self) -> None:
        """Initializes the environment's action space."""

        self.action_space = spaces.Dict({
            # First, for each pending request, indicate whether we reject it.
            "req_rejections": spaces.MultiBinary(self.num_pending_requests),
            # Second, for each pending request, indicate which vehicle to assign to it
            # No-assignment is indicated by setting values to self.num_vehicles
            "req_assgts": spaces.MultiDiscrete(np.full((self.num_pending_requests,), fill_value=self._num_vehicles + 1, dtype=np.int)),
            # Third, for each vehicle, a repositioning instruction.
            # No-assignment is indicated by setting values to self.num_lots
            "reposition": spaces.MultiDiscrete(np.full((self._num_vehicles,), fill_value=self.num_lots + 1, dtype=np.int)),
        })

    
    
    def _set_action_space(self) -> None:
        """Updates the environment's action space."""

        # Update the spaces that depend on the number of pending requests.

        # Check to confirm that we need to update the action space's shape
        if self.action_space.spaces["req_rejections"].shape != (self.num_pending_requests,):

            self.action_space.spaces["req_rejections"] = spaces.MultiBinary(self.num_pending_requests)

            self.action_space.spaces["req_assgts"] = spaces.MultiDiscrete(
                np.full((self.num_pending_requests,), fill_value=self._num_vehicles + 1, dtype=np.int)
            )
        

    def _make_observation_space(self) -> gym.Space:
        """Initializes the environment's observation space."""
        
        self.observation_space = spaces.Dict({

            # First element is the time, a Box of shape (1,) with value bound by 0, MAX_TIME
            "time": spaces.Box(
                low=0,
                high=self._MAX_TIME,
                shape=(1,),
                dtype=np.float64
            ),

            # Second element is the day of the week
            "dow": spaces.Discrete(7),

            # Third element indicates the locations of pending requests
            "request_locs": spaces.Box(
                low=(
                    np.tile([self.x_range[0], self.y_range[0]], self.num_pending_requests * 2)
                    .reshape(self.num_pending_requests,2,2)
                ),
                high=(
                    np.tile([self.x_range[1], self.y_range[1]], self.num_pending_requests * 2)
                    .reshape(self.num_pending_requests,2,2)
                ),
                dtype=np.float64
            ),

            # Fourth element indicates the release time of the pending requests
            "request_times": spaces.Box(
                low=0,
                high=self._MAX_TIME,
                shape=(self.num_pending_requests,),
                dtype=np.float64
            ),

            # Fifth, vehicle locations, a Box of shape (V, 2)
            #
            # [[v0x, v0y], [v1x, v1y], ... [vVx, vVy]]
            # Values bounded by x and y ranges
            "v_locs": spaces.Box(
                low=np.tile([self.x_range[0], self.y_range[0]], self._V).reshape(self._V, 2),
                high=np.tile([self.x_range[1], self.y_range[1]], self._V).reshape(self._V, 2),
                dtype=np.float64
            ),

            # Sixth element contains vehicles' job types
            #
            # [[v0j0m, v0j1m, v0j2m], [v1j0m, v1j1m, v1j2m], ... [vVj0m, vVj1m, vVj2m]]
            # Values are those for Jobs: 0, 1, 2, 3, 4 (idle, reposition, setup, process, and null)
            "v_jobs": spaces.MultiDiscrete(np.full((self._V, 3), len(Jobs))),

            # Seventh element are vehicles' jobs' locations:
            #
            # Shape is (num_vehicles, MAX_JOBS, 2 (origin/destination), 2 (x/y))
            # Values are all bounded by x and y ranges
            "v_job_locs": spaces.Box(
                low=(
                    np.tile([self.x_range[0], self.y_range[0]], self._num_vehicles * 3 * 2)
                    .reshape(self._num_vehicles, 3, 2, 2)
                ),
                high=(
                    np.tile([self.x_range[1], self.y_range[1]], self._num_vehicles * 3 * 2)
                    .reshape(self._num_vehicles, 3, 2, 2)
                ),
                dtype=np.float64),
        })

    def _set_observation_space(self) -> None:
        """Updates the environment's observation space."""

        # Update the spaces that depend on the number of pending requests.

        # Check to confirm that we need to update the observation space's shape
        if self.observation_space.spaces["request_times"].shape != (self.num_pending_requests,):

            self.observation_space.spaces["request_locs"] = spaces.Box(
                low=(
                    np.tile([self.x_range[0], self.y_range[0]], self.num_pending_requests * 2)
                    .reshape(self.num_pending_requests,2,2)
                ),
                high=(
                    np.tile([self.x_range[1], self.y_range[1]], self.num_pending_requests * 2)
                    .reshape(self.num_pending_requests,2,2)
                ),
                dtype=np.float64
            )

            self.observation_space.spaces["request_times"] = spaces.Box(
                low=0,
                high=self._MAX_TIME,
                shape=(self.num_pending_requests,),
                dtype=np.float64
            )


    def xy_to_zone(self, xys: np.ndarray) -> np.ndarray:
        """Converts an array of XY coordinates into an array of taxi zone IDs."""

        if xys.shape[-1] != 2:
            raise ValueError("xys last dimension should be 2 ([x,y]).")

        return self._geom.xy_to_zone(xys)
    
    
    def _get_request_value(self, origin, destination):
        """ Returns the value of a request based on its origin and destination."""

        dist = self._geom.dist(origin, destination, pairwise=False)
        return self._FIXED_REWARD + self._VARIABLE_REWARD_DIST * dist


    def _get_curr_hr(self) -> int:
        """Returns the hour of the current time."""

        # Time is tracked in seconds starting from _EPS_START_HR. Convert that to hours elapsed, add to our
        # start time, then roll it around if necessary.
        return self._get_hr_from_time(self.time)
    
    
    def _action_was_slow(self) -> bool:
        """Checks if an agent was too slow providing an action."""

        curr_time = time.time()

        # An action was slow if a previous release time was defined, and if
        # the time since then exceeds the action timelimit.
        return (
            self._obs_release_time is not None
            and curr_time - self._obs_release_time > self._action_timelimit
        )
    
    
    def _get_hr_from_time(self, time: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Returns the hour of the day (as an integer) from the elapsed time."""
        return np.round(((time // self._S_PER_HR) + self._EPS_START_HR) % self._HRS_PER_DAY).astype(int)
    
    
    def _make_null_jobs(self, mask: np.ndarray, jobs: Union[int, List[int]]) -> None:
        """Marks jobs as null for the vehicle slice/mask given by mask."""

        if isinstance(jobs, int):
            jobs = [jobs]

        xcols = [f"j{n}{od}x" for n in jobs for od in ("o", "d")]
        ycols = [f"j{n}{od}y" for n in jobs for od in ("o", "d")]
        tcols = [f"j{n}{od}t" for n in jobs for od in ("o", "d")]

        self._vehicles.loc[mask, xcols] = self._NULL_X
        self._vehicles.loc[mask, ycols] = self._NULL_Y
        self._vehicles.loc[mask, tcols] = self._NEVER

        mcols = [f"j{n}m" for n in jobs]
        self._vehicles.loc[mask, mcols] = Jobs.NULL
    
    
    def _shift_n_jobs(self, mask:np.ndarray, n:int) -> None:
        """For the vehicles specified in `mask`, perform n job shifts."""

        orig_dtypes = self._vehicles.dtypes

        if n == 1:
            self._vehicles.loc[mask, ["j1m", "j1ox", "j1oy", "j1ot", "j1dx", "j1dy", "j1dt"]] = (
                self._vehicles.loc[mask, ["j2m", "j2ox", "j2oy", "j2ot", "j2dx", "j2dy", "j2dt"]].to_numpy()
            )
            self._vehicles.loc[mask, ["j2m", "j2ox", "j2oy", "j2ot", "j2dx", "j2dy", "j2dt"]] = (
                self._vehicles.loc[mask, ["j3m", "j3ox", "j3oy", "j3ot", "j3dx", "j3dy", "j3dt"]].to_numpy()
            )
            self._make_null_jobs(mask, 3)

        elif n == 2:
            self._vehicles.loc[mask, ["j1m", "j1ox", "j1oy", "j1ot", "j1dx", "j1dy", "j1dt"]] = (
                self._vehicles.loc[mask, ["j3m", "j3ox", "j3oy", "j3ot", "j3dx", "j3dy", "j3dt"]].to_numpy()
            )
            self._make_null_jobs(mask, [2,3])

        elif n == 3:
            self._make_null_jobs(mask, [1,2,3])
        
        else:
            raise ValueError("Invalid value specified for 'n'")

        # Columns have been shifted. Ensure dtypes are preserved
        self._vehicles = self._vehicles.astype(orig_dtypes)
    
    
    def _check_valid_action(self, action) -> None:
        """Ensures that `action` is valid."""

        # 0) Make sure that the action fits the action space definition
        if not self.action_space.contains(action):
            raise ValueError("Invalid action provided. Not contained in action space.")
        
        # 1) Make sure that no vehicle was assigned to more than one request.
        actual_assgts = action["req_assgts"][action["req_assgts"] != self._V]
        if len(np.unique(actual_assgts)) != len(actual_assgts):
            raise ValueError("Invalid 'req_assgts' provided. A vehicle was assigned to more than one request.")


    def _check_assignment_feasibility(self, vs: pd.DataFrame, reqs: pd.DataFrame) -> np.ndarray:
        """Returns a boolean array indicating whether the vehicles ('vs') can be assigned to the requests ('reqs')."""

        assert len(vs) == len(reqs), "vs and reqs should be of the same length."

        # If there's nothing to check, return an empty boolean array
        if len(vs) == 0:
            return np.full((0,), fill_value=True, dtype=np.bool)

        # Get the mean travel times from vehicles' next available location to requests' origins
        travel_times = self._geom.travel_time(
            o=vs[["avail_x", "avail_y"]].to_numpy(),
            d=reqs[["ox", "oy"]].to_numpy(),
            hr=self._get_curr_hr(),
            mean=True,
            pairwise=False,
        )

        # The amount of time the customer has to wait is this travel time, plus the additional time until the
        # vehicle can depart.
        cust_waits = travel_times + (vs["avail_t"].to_numpy() - self.time)

        # It's feasible where the expected waiting time is less than the max waiting time
        time_feasible = cust_waits <= self._MAX_WAIT

        # Also ensure that vehicles' jobs don't preclude them from taking on the assignment:
        # Vehicles can't have a second non-preemptable job already lined up
        job_feasible = np.isin(vs["j2m"].to_numpy(), (Jobs.IDLE, Jobs.REPOSITION, Jobs.NULL))

        # Done.
        return time_feasible & job_feasible
    
    
    def _get_server_job_col_updates(self, req_idxs, v_idxs) -> np.ndarray:
        """Provides serving vehicles' new job descriptions that can be inserted into self._vehicles.
        
        Inputs:
        
            req_idxs: The indices of the requests that are being served.
            
            v_idxs: The indices of the vehicles that are serving them.
            
        Outputs:
        
            Numpy array with a row for each serving vehicle, with columns for the fields that
            need to be updated in self._vehicles.
            
        """

        # The first new job for the serving vehicles is to travel from their next available
        # location to the origin of the new job.
        first_job_oloc = self._vehicles.loc[v_idxs, ["avail_x", "avail_y"]].to_numpy()
        first_job_dloc = self._requests.loc[req_idxs, ["ox", "oy"]].to_numpy()
        first_job_ot = self._vehicles.loc[v_idxs, "avail_t"].to_numpy()
        first_job_dt = first_job_ot + self._geom.travel_time(
            o=first_job_oloc,
            d=first_job_dloc,
            hr=self._get_hr_from_time(first_job_ot),
            mean=False,
            pairwise=False,
        )

        # The next job for the serving vehicles is to serve this new customer.
        second_job_oloc = self._requests.loc[req_idxs, ["ox", "oy"]].to_numpy()
        second_job_dloc = self._requests.loc[req_idxs, ["dx", "dy"]].to_numpy()
        second_job_ot = first_job_dt
        second_job_dt = second_job_ot + self._requests.loc[req_idxs, "pt"].to_numpy()

        # The serving vehicles will next be available where/when they drop off the request.
        # That is also the next time at which they will trigger a new epoch
        avail_loc = second_job_dloc
        avail_t = second_job_dt
        epoch_t = avail_t

        # Lump all of these job-description columns into one horizontal chunk that can be inserted into
        # the appropriate columns in self._vehicles.
        first_job_hstack = np.hstack([
            first_job_oloc,
            first_job_ot.reshape(-1,1),
            first_job_dloc,
            first_job_dt.reshape(-1,1)
        ])
        second_job_hstack = np.hstack([
            second_job_oloc,
            second_job_ot.reshape(-1,1),
            second_job_dloc,
            second_job_dt.reshape(-1,1)
        ])
        full_hstack = np.hstack([
            first_job_hstack,
            second_job_hstack,
            avail_loc,
            avail_t.reshape(-1,1),
            epoch_t.reshape(-1,1)
        ])

        return full_hstack


    def _update_servers_job_cols(self, req_idxs, v_idxs) -> None:
        """Updates the job description columns in self._vehicles for those vehicles given
        a new request assignment.
        
        Inputs:
        
            req_idxs: The indices of the requests that are being served.
            
            v_idxs: The indices of the vehicles that are serving them.
            
        Outputs:
        
            None. Just updates self._vehicles
            
        """

        # If there are no requests being served, we can just return now.
        if len(req_idxs) == 0:
            return

        # Get the data to populate the serving vehicles' job columns
        job_col_updates = self._get_server_job_col_updates(req_idxs, v_idxs)

        # Note whether servers are currently busy
        server_busy_mask = (self._vehicles['avail_t'] != self.time)[v_idxs]
        
        # The servers that are currently busy
        busy_servers = v_idxs[server_busy_mask]
        # And the ones that are not
        now_servers = v_idxs[~server_busy_mask]

        #
        ## First, the busy servers.
        ## We will be populating their second and third jobs (their first job is the
        ## service of an existing customer).
        #

        # Define the job types of these new jobs
        self._vehicles.loc[busy_servers, ["j2m", "j3m"]] = Jobs.SETUP, Jobs.PROCESS

        # We are going to update the x/y/t fields for the origins and destinations of the 2nd and 3rd jobs
        update_cols = (
            [f"j{j_idx}{od}{xyt}" for j_idx in (2,3) for od in ("o", "d") for xyt in ("x", "y", "t")]
            + ["avail_x", "avail_y", "avail_t", "epoch_t"]
        )
        
        # Perform the update
        self._vehicles.loc[busy_servers, update_cols] = job_col_updates[server_busy_mask]

        #
        ## Next, the non-busy servers.
        ## Because these vehicles can begin service immediately, we update their
        ## first and second jobs. Their third job will be null.
        #

        # Again define the types of these new jobs.
        self._vehicles.loc[now_servers, ["j1m", "j2m"]] = Jobs.SETUP, Jobs.PROCESS

        # Going to update the x/y/t fields for the origins and destinations of the 1st and 2nd jobs
        update_cols = (
            [f"j{j_idx}{od}{xyt}" for j_idx in (1,2) for od in ("o", "d") for xyt in ("x", "y", "t")]
            + ["avail_x", "avail_y", "avail_t", "epoch_t"]
        )

        # Perform the update
        self._vehicles.loc[now_servers, update_cols] = job_col_updates[~server_busy_mask]

        # Filling in a third null job
        self._make_null_jobs(mask=now_servers, jobs=3)

        return
    
    def _update_repos_job_cols(self, repos_v_idxs, repos_lots) -> None:
        """Updates the job description columns in self._vehicles for those vehicles given
        a new repositioning instruction.

        Note: 
        
        Inputs:
        
            repos_v_idxs: The indices of the vehicles that were given repositioning instructions.
            
            repos_lots: The lots to which those vehicles were told to reposition.
            
        Outputs:
        
            None. Just updates self._vehicles
            
        """

        # If no vehicles received a repositioning instruction, we can just return.
        if len(repos_v_idxs) == 0:
            return

        # First, get the data to populate the repositioning vehicles' job columns.
        # Their first new job is the repositioning job.
        first_job_oloc = self._vehicles.loc[repos_v_idxs, ["x", "y"]].to_numpy()
        first_job_dloc = repos_lots[["x", "y"]].to_numpy()
        first_job_ot = self.time
        first_job_dt = first_job_ot + self._geom.travel_time(
            o=first_job_oloc,
            d=first_job_dloc,
            hr=self._get_hr_from_time(first_job_ot),
            mean=False,
            pairwise=False,
        )

        # After repositioning, these vehicles will then idle at the lot to which they were assigned.
        second_job_oloc = first_job_dloc
        second_job_dloc = second_job_oloc
        second_job_ot = first_job_dt
        second_job_dt = self._NEVER

        # Where and when these vehicles will be next available is not known yet -- this will depend on when
        # the next epoch is triggered.
        avail_loc = second_job_dloc
        avail_t = self.time

        # These vehicles will not trigger a new epoch
        epoch_t = self._NEVER

        # We now combine these job updates into a single block that can be inserted into self._vehicles
        first_job_hstack = np.hstack([
            first_job_oloc,
            np.full_like(first_job_dt, first_job_ot).reshape(-1,1),
            first_job_dloc,
            first_job_dt.reshape(-1,1)
        ])
        second_job_hstack = np.hstack([
            second_job_oloc,
            second_job_ot.reshape(-1,1),
            second_job_dloc,
            np.full_like(second_job_ot, second_job_dt).reshape(-1,1)
        ])
        full_hstack = np.hstack([
            first_job_hstack,
            second_job_hstack,
            avail_loc,
            np.full_like(first_job_dt, avail_t).reshape(-1,1),
            np.full_like(second_job_ot, epoch_t).reshape(-1,1)
        ])

        # Begin updating self._vehicles. First, the jobs' types
        self._vehicles.loc[repos_v_idxs, ["j1m", "j2m"]] = Jobs.REPOSITION, Jobs.IDLE

        # Next, going to update the x/y/t fields for the origins and destinations of the 1st and 2nd jobs.
        # Get a list of those columns.
        update_cols = (
            [f"j{j_idx}{od}{xyt}" for j_idx in (1,2) for od in ("o", "d") for xyt in ("x", "y", "t")]
            + ["avail_x", "avail_y", "avail_t", "epoch_t"]
        )

        # Fill the columns.
        self._vehicles.loc[repos_v_idxs, update_cols] = full_hstack
        
        # Lastly, give these vehicles a third null job.
        self._make_null_jobs(mask=repos_v_idxs, jobs=3)

        return


    def _get_jobs_dists(self) -> np.ndarray:
        """Returns a Vx3 array of the distances covered in vehicles' jobs."""

        job_dists = np.hstack([
            # The distance they have to travel for their first jobs...
            self._geom.dist(
                o=self._vehicles[["j1ox", "j1oy"]].to_numpy(),
                d=self._vehicles[["j1dx", "j1dy"]].to_numpy(),
                pairwise=False
            ).reshape(-1,1),
            # For their second jobs...
            self._geom.dist(
                o=self._vehicles[["j2ox", "j2oy"]].to_numpy(),
                d=self._vehicles[["j2dx", "j2dy"]].to_numpy(),
                pairwise=False
            ).reshape(-1,1),
            # And for their third jobs...
            self._geom.dist(
                o=self._vehicles[["j3ox", "j3oy"]].to_numpy(),
                d=self._vehicles[["j3dx", "j3dy"]].to_numpy(),
                pairwise=False
            ).reshape(-1,1)
        ])

        return job_dists
   
   
    def _check_null_v_assgts(self, serving_idxs, repos_idxs) -> None:
        """Makes sure that any vehicle without an instruction has received one."""

        needs_job = self._vehicles.index[self._vehicles["j1m"] == Jobs.NULL]

        got_job = needs_job.isin(serving_idxs) | needs_job.isin(repos_idxs)

        if np.any(~got_job):
            raise ValueError(
                f"The following vehicles requiring an instruction did not receive one:\n{needs_job[~got_job]}"
            )
        
        return

    
    def step(self, action):
        """Advance the environment from the current state to the next state, dependent on action.

          Inputs:

            action: an action provided by the agent
          
          Outputs:

            A tuple with the following:
                
                observation: agent's observation of the environment's new state
                reward: amount of reward accrued for having taken action
                done: whether the episode has ended
                info: dictionary with auxiliary diagnostic information (currently always empty)

        """

        # IDEA should agents incur a penalty for the time that requests have to wait before assignment?
        # (if they sit pending for awhile)

        # Make sure the user wasn't too slow to provid an action
        if self._action_was_slow():
            logging.warning(f"Took too long to provide action.")
            return None, -np.inf, True, {}

        # Make sure the action was valid.
        self._check_valid_action(action)
        
        # Initialize the new reward accrued
        reward = 0.0

        # Get the subset of requests that are currently available for assignment
        reqs = self._get_pending_requests()
        reqs_idx = reqs.index
        
        #
        ## Phase I: Updating requests & vehicles' schedules based on the specified action.
        #

        ## Phase I.I: For the service of requests.

        # Update the assignments to take rejections into account
        req_assgts = np.where(action['req_rejections'], self._V, action['req_assgts'])
        if any(req_assgts != action['req_assgts']):
            logging.warning("Some assignments were ignored bc the request was rejected.")
        
        # Check the feasibility of the assignments
        assgd_reqs = reqs.loc[req_assgts != self._V, :]
        serving_vs = self._vehicles.loc[req_assgts[req_assgts != self._V], :]
        infeasible = ~(self._check_assignment_feasibility(serving_vs, assgd_reqs))
        
        # If any of the assignments are infeasible...
        if any(infeasible):

            # Note which vehicles had bad assignments and warn the user
            bad_vs = serving_vs.index[infeasible]
            logging.warning(f"Ignoring infeasible assignments for the following vehicles:\n{bad_vs}.")
            
            # Update req_assgts to ignore the assignments for the requests that these vehicles
            # were assigned to.
            req_assgts = np.where(np.isin(req_assgts, bad_vs), self._V, req_assgts)

        # Note the indices of the requests that have been rejected
        rejected_reqs = reqs_idx[action["req_rejections"].astype(np.bool)]

        # A mask to indicate which of the pending requests received an assignment
        assgd_reqs_mask = req_assgts != self._V

        # The indices of the requests that received an assignment
        assgd_reqs_idxs = reqs_idx[assgd_reqs_mask]

        # The indices of the vehicles that are serving the assigned requests
        serving_idxs = req_assgts[assgd_reqs_mask]

        ## Phase I.I.a) Updating the vehicles' df for the service of the new requests.

        self._update_servers_job_cols(assgd_reqs_idxs, serving_idxs)

        ## Phase I.I.b) Updating the requests df

        # The requests that were rejected we can now mark as such. Their vehicle column will remain None.
        self._requests.loc[rejected_reqs, "rejected"] = True
        
        # For the requests that were assigned to vehicles, update their vehicle column.
        self._requests.loc[assgd_reqs_idxs, "vehicle"] = serving_idxs

        ## Phase I.I complete.
        # Accumulate reward for the requests that were newly assigned to vehicles
        reward += self._requests["value"][assgd_reqs_idxs].sum()

        ## Phase I.II: Update vehicles' schedules following from the specified repositioning actions.

        reposs = action["reposition"]
        has_repos_assgt = reposs != self._D

        # Note if there are any bad repositioning assignments (those given to busy vehicles).
        # ("busy vehicles" here also includes those that were just now assigned to serve a request)
        bad_repos_assgt = has_repos_assgt & (self._vehicles['avail_t'] != self.time).to_numpy()
        if np.any(bad_repos_assgt):
            logging.warning(
                f"The following vehicles are busy serving but were given repositioning assignments:\n"
                f"{self._vehicles.index[bad_repos_assgt]}"
            )
            # Kill these repositioning assignments and update our mask for the repositioning vehicles
            reposs[bad_repos_assgt] = self._D
            has_repos_assgt = reposs != self._D
        
        # The indices of the vehicles that received a repositioning assignment
        repos_v_idxs = self._vehicles.index[has_repos_assgt]
        
        # Make sure that any vehicle that is currently idle either received a repositioning
        # assignment or a service assignment.
        self._check_null_v_assgts(serving_idxs, repos_v_idxs)

        # The lots that they were told to reposition to
        repos_lots = self._lots.loc[reposs[has_repos_assgt], :]

        # Update the repositioning vehicles' job columns accordingly.
        self._update_repos_job_cols(repos_v_idxs, repos_lots)

        ## PHASE I.III: Determining when the next epoch will occur and why

        # i) The next time at which an EV will trigger an epoch
        next_ev_epoch_time = self._vehicles["epoch_t"].min()
        # ii) The next time at which a new request will trigger an epoch
        next_req_epoch_time = self._requests.at[self._next_request_idx, "time"]
        # iii) The next time at which we force an epoch to trigger:
        # - either the end of the episode or the next time requested by the user
        next_forced_epoch_time = min(self._MAX_TIME, self.time + self._max_interdecision_time)
        # The next epoch occurs at the earliest of these events.
        next_epoch_time = min(next_ev_epoch_time, next_req_epoch_time, next_forced_epoch_time)
        is_new_request = next_epoch_time == next_req_epoch_time

        #
        ## Phase II: Updating vehicles' schedules to be "current", based on the next epoch time.
        #

        ## Phase II.I: Computing the percentage of each job that has been completed for each vehicle,
        ##   and using that to compute the travel cost for the action and the number of jobs that
        ##   should be removed from each vehicle's job list.

        job_otimes = self._vehicles[["j1ot", "j2ot", "j3ot"]].to_numpy()
        job_dtimes = self._vehicles[["j1dt", "j2dt", "j3dt"]].to_numpy()
        job_durations = job_dtimes - job_otimes
        job_durations_inf_zeros = np.where(job_durations == 0, np.inf, job_durations)  # so we can perform valid division
        job_pct_remaining_now = np.clip(job_dtimes - self.time, 0, job_durations) / job_durations_inf_zeros
        job_pct_remaining_next = np.clip(job_dtimes - next_epoch_time, 0, job_durations) / job_durations_inf_zeros
        job_pct_completed = job_pct_remaining_now - job_pct_remaining_next
        assert np.all(np.isfinite(job_pct_completed)), "Values should all be finite at this point."
        job_dists = self._get_jobs_dists()
        dists_traveled = job_pct_completed * job_dists
        reward -= np.sum(dists_traveled) * self._TRAVEL_COST

        # How many jobs will be completed by the time of the next epoch?
        num_to_remove = (next_epoch_time >= job_dtimes).sum(axis=1)

        ## Phase II.II: Final updates to vehicles
        
        ## Phase II.II.a) Shift jobs to remove the completed ones.

        for n in (1,2,3):
            if any(num_to_remove == n):
                self._shift_n_jobs(mask=num_to_remove == n, n=n)

        ## Phase II.II.b) For vehicles who just finished a job (triggering the next epoch),
        ##   their current locations (x,y cols) are equal to the locations they said they
        ##   would next be available.

        # These vehicles will have a null job in their first position
        finishing_vs = self._vehicles["j1m"] == Jobs.NULL

        # A quick self-check that jobs have been updated as expected.
        assert all(finishing_vs == (self._vehicles["avail_t"] == next_epoch_time)), (
            "Any vehicle without a first job should have triggered the next epoch."
        )

        # Set these vehicles' current locations.
        self._vehicles.loc[finishing_vs, ["x", "y"]] = self._vehicles.loc[finishing_vs, ["avail_x", "avail_y"]].to_numpy()

        ## Phase II.II.c) Get other vehicles' new current locations (their x,y cols)

        # We call the non-finishing vehicles "progress" vehicles, meaning they are simply making progress
        progress_vs = ~finishing_vs

        # We know they are on their first jobs, so compute the percentage of that job that has been completed
        rel_elapsed_ts = (
            (next_epoch_time - self._vehicles.loc[progress_vs, "j1ot"]).to_numpy()
            / (self._vehicles.loc[progress_vs, "j1dt"] - self._vehicles.loc[progress_vs, "j1ot"]).to_numpy()
        )
        assert np.all(np.isfinite(rel_elapsed_ts)), "All first jobs should have finite elapsed times (non-zero duration)."

        # Their location is at that percentage of the difference between their first job's origin and destination
        self._vehicles.loc[progress_vs, ["x", "y"]] = (
            # The first job's origin...
            self._vehicles.loc[progress_vs, ["j1ox", "j1oy"]].to_numpy()
            + (
                # ... plus (the percentage of the job that's been completed...
                rel_elapsed_ts[:, None]
                # ... times the distance it's supposed to travel to complete the job).
                * (self._vehicles.loc[progress_vs, ["j1dx", "j1dy"]].to_numpy() - self._vehicles.loc[progress_vs, ["j1ox", "j1oy"]].to_numpy())
            )
        )

        ## Phase II.II.d) Additional considerations for any non-serving vehicle...
        ##   Their next-available time and location are now and wherever they currently are, respectively.
        not_busy = ~self._vehicles["j1m"].isin((Jobs.SETUP, Jobs.PROCESS))
        self._vehicles.loc[not_busy, ["avail_x", "avail_y"]] = self._vehicles.loc[not_busy, ["x", "y"]].to_numpy()
        self._vehicles.loc[not_busy, "avail_t"] = next_epoch_time
        
        ## Phase III: Updating requests and other environment objects for the next epoch
        
        # Note the new time
        self.time = next_epoch_time

        # Update request-related things
        self._is_new_request = is_new_request
        if self._is_new_request:
            # Release the new request
            self._requests.at[self._next_request_idx, "released"] = True
            # Update the index of the next request that will pop up
            self._next_request_idx += 1

        # Purge (reject) any pending requests that have been around longer than the MAX_WAIT time
        self._requests.loc[
            self._requests["vehicle"].isna() & (self.time - self._requests["time"] > self._MAX_WAIT),
            "rejected"
        ] = True

        # Update the total accumulated rewards
        self.rewards += reward

        # Set the number of assignable requests and update the action and observation spaces
        self.num_pending_requests = self._get_num_pending_requests()
        self._set_action_space()
        self._set_observation_space()
        
        # Create the observation to send to the agent
        obs = self._make_state()

        # Note the time at which this observation is being released to the user.
        self._obs_release_time = time.time()

        return obs, reward, self.time >= self._MAX_TIME, {}


    def reset(self):
        """ Sets the environment to an initial state and returns that state."""

        # Increment episode counter
        self.curr_episode += 1

        # Reset step count
        self.curr_step = 0

        # Reset the clock and score counter
        self.time = 0.0
        self.rewards = 0.0
        
        # Update seeds
        self._reseed()

        # Initialize vehicles for the episode
        self._initialize_vehicles()

        # Initialize the episode's requests
        self._generate_requests()
        
        # Note that there is no current request and update the action and observation spaces
        self._is_new_request = False
        self.num_pending_requests = 0
        self._set_observation_space()
        self._set_action_space()

        # And that the next request to arise will be the first (zeroth) request.
        self._next_request_idx = 0

        # Generate the initial state
        obs = self._make_state()

        # We don't do a time check on the first action (may be more external computation required after the reset)
        self._obs_release_time = None

        return obs

    
    def _make_state(self):
        """Create the current agent-facing state."""

        curr_requests = self._get_pending_requests()

        obs = {

            # Episode time
            "time": self.time,

            # Day of the week
            "dow": self._dow,

            # Where are the pending requests
            "request_locs": (
                curr_requests.loc[:, ["ox", "oy", "dx", "dy"]]
                .to_numpy()
                .reshape(self.num_pending_requests, 2, 2)
                .astype(np.float64)
            ),

            # When were the pending requests released
            "request_times": (
                curr_requests.loc[:, "time"]
                .to_numpy()
                .reshape(self.num_pending_requests, )
                .astype(np.float64)
            ),

            # Vehicle locations
            # [[v0x, v0y], [v1x, v1y], ... [vVx, vVy]]
            "v_locs": (self._vehicles[["x", "y"]].to_numpy().astype(np.float64)),

            # Vehicles' job types
            # [[v0j1m, v0j2m, v0j3m], [v1j1m, v1j2m, v1j3m], ... [vVj1m, vVj2m, vVj3m]]
            "v_jobs": (
                self._vehicles[["j1m", "j2m", "j3m"]]
                .to_numpy().astype(np.int8)
            ),

            # Vehicles' jobs' locations:
            # Shape is (num_vehicles, 3, 2 (origin/destination), 2 (x/y))
            "v_job_locs": (
                self._vehicles[[
                    "j1ox", "j1oy", "j1dx", "j1dy",
                    "j2ox", "j2oy", "j2dx", "j3dy",
                    "j3ox", "j3oy", "j3dx", "j3dy"
                ]].to_numpy().reshape(self._V, 3, 2, 2).astype(np.float64)
            ),
        }

        return obs


    def _get_assgd_req_locs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns arrays for the origins and destinations of already-assigned requests.

        If there are N already-assigned requests, each array has shape (N, 2). The
        0th column contains x coordinates and the 1st column contains y coordinates.
        
        """

        assgd_reqs = np.vstack([
            self._vehicles.loc[
                self._vehicles[f"j{j_idx}m"] == Jobs.PROCESS,
                [f"j{j_idx}ox", f"j{j_idx}oy", f"j{j_idx}dx", f"j{j_idx}dy"]
            ].values
            for j_idx in (1,2,3)
        ])
        
        return assgd_reqs[:, :2], assgd_reqs[:, -2:]


    def _get_pending_req_locs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns arrays for the origins and destinations of unassigned requests.

        If there are N unassigned requests, each array has shape (N, 2). The
        0th column contains x coordinates and the 1st column contains y coordinates.
        
        """

        pending_reqs = self._get_pending_requests()
        
        return pending_reqs[["ox", "oy"]].to_numpy(), pending_reqs[["dx", "dy"]].to_numpy()
    
    
    def _prep_rendering(self) -> None:

        self._fig, self._axes = plt.subplots()

        # Define axes' bounds, hide grid and ticks, and set the axes to use identical scales
        self._axes.set_xlim(self.x_range)
        self._axes.set_ylim(self.y_range)
        self._axes.axis('off')
        self._axes.set_aspect('equal')

        # Plot the taxi zones
        self._plt_zonepolys = self._axes.add_collection(self._geom.zone_polys)

        # Plot the lots
        self._plt_lots = self._axes.scatter(
            x=self._lots['x'].values,
            y=self._lots['y'].values,
            marker='s',
            c="deepskyblue",
            label="lot",
            alpha=0.5
        )
    
    
    def render(self, mode='human', close=False):

        raise NotImplementedError("Rendering not yet supported...")

        # fig, axes = plt.subplots()

        # # Define axes' bounds, hide grid and ticks, and set the axes to use identical scales
        # axes.set_xlim(self.x_range)
        # axes.set_ylim(self.y_range)
        # axes.axis('off')
        # axes.set_aspect('equal')

        # # Plot the taxi zones
        # axes.add_collection(self._geom.zone_polys)

        # # Plot the lots
        # axes.scatter(
        #     x=self._lots['x'].values,
        #     y=self._lots['y'].values,
        #     marker='s',
        #     c="deepskyblue",
        #     label="lot",
        #     alpha=0.5
        # )

        # Plot the requests that have been assigned but are not yet completed
        req_os, req_ds = self._get_assgd_req_locs()
        # Plot the origins (o's)...
        plt_assgd_req_os = self._axes.scatter(x=req_os[:, 0], y=req_os[:, 1], marker='o', facecolors='none', edgecolors='lightpink')
        # ...  and destinations (x's).
        plt_assgd_req_ds = self._axes.scatter(x=req_ds[:, 0], y=req_ds[:, 1], marker='x', c='lightpink')
        # And connect them
        plt_assgd_req_cnxns = self._axes.plot(
            # Our x and y arrays have a column for each line (request) and a row for each pt in the line (its origin and destination)
            np.vstack([req_os[:, 0], req_ds[:, 0]]),
            np.vstack([req_os[:, 1], req_ds[:, 1]]), 
            c='lightpink',
        )

        # Plot the pending requests (active but unassigned).
        req_os, req_ds = self._get_pending_req_locs()
        # Plot the origins (o's)...
        plt_pending_req_os = self._axes.scatter(x=req_os[:, 0], y=req_os[:, 1], marker='o', facecolors='none', edgecolors='r', label="Request origin")
        # ...  and destinations (x's).
        plt_pending_req_ds = self._axes.scatter(x=req_ds[:, 0], y=req_ds[:, 1], marker='x', c='red')
        # And connect them
        plt_pending_req_cnxns = self._axes.plot(
            np.vstack([req_os[:, 0], req_ds[:, 0]]),
            np.vstack([req_os[:, 1], req_ds[:, 1]]), 
            c='r',
        )

        # Plot the vehicles
        plt_vehicles = self._axes.scatter(
            x=self._vehicles["x"].values,
            y=self._vehicles["y"].values,
            label='vehicle',
            color='gold',
            edgecolors='silver',
            marker='o'
        )
        
        # Put a legend below x axis
        plt_legend = self._axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=3)

        # Add a title with some info about the current state
        # TODO sampled_starttime is no more.
        # Update to just show a day of the week
        time = self._sampled_starttime + datetime.timedelta(seconds=self.time)
        time_str = datetime.datetime.strftime(time, self._TIME_STR_FORMAT_PRINT)
        status_string = f"{time_str}\nReward: ${self.rewards:.2f}"

        # If episode is over, add this to the title info
        if self.time >= self._MAX_TIME:
            status_string = f"TERMINAL STATE\n{status_string}"

        # Stick it on the plot
        self._axes.set_title(status_string)
        
        # Render the canvas and get it as an rgbarray to return to the user
        self._fig.canvas.draw_idle()
        rgb_array = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_array = rgb_array.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))

        # We have the RGB array we need. Remove the temporary graph elements we created here
        temp_plt_objs = [
            plt_assgd_req_os,
            plt_assgd_req_ds,
            plt_assgd_req_cnxns,
            plt_pending_req_os,
            plt_pending_req_ds,
            plt_pending_req_cnxns,
            plt_vehicles,
            plt_legend
        ]

        for plt_obj in temp_plt_objs:
            # The request connections are lists of lines.
            # We want to remove each of those lines from our axes.
            if isinstance(plt_obj, list):
                for _ in range(len(plt_obj)):
                    line = plt_obj.pop()
                    self._axes.lines.remove(line)
                    del line
            else:
                plt_obj.remove()

        return rgb_array


    def _initialize_vehicles(self) -> None:
        """Initializes the environment's vehicles at random locations."""

        # Sample initial lots for each vehicle
        init_lots = self._vehicle_sampler.choice(
            self._lots.index,
            size=self._num_vehicles,
            p=self._lots["init_weight"]
        )

        # Initializing our vehicles by giving them initial x/y locs at these lots
        vehicles = self._lots.loc[init_lots, ["x", "y"]].copy().reset_index(drop=True)

        # All vehicles have an initial idling job
        vehicles["j1m"] = Jobs.IDLE

        # These jobs' origins and destinations are equivalent to their initial locations
        vehicles[["j1ox", "j1oy"]] = vehicles[["x", "y"]]
        vehicles[["j1dx", "j1dy"]] = vehicles[["x", "y"]]

        # The jobs begin now...
        vehicles[["j1ot"]] = self.time
        # And have no end.
        vehicles[["j1dt"]] = self._NEVER

        # The remaining jobs are all null
        for job_type_col in ["j2m", "j3m"]:
            vehicles[job_type_col] = Jobs.NULL

        # With null locations and times.
        for null_x_col in ["j2ox", "j2dx", "j3ox", "j3dx"]:
            vehicles[null_x_col] = self._NULL_X
        for null_y_col in ["j2oy", "j2dy", "j3oy", "j3dy"]:
            vehicles[null_y_col] = self._NULL_Y
        for null_time_col in ["j2ot", "j2dt", "j3ot", "j3dt"]:
            vehicles[null_time_col] = self._NEVER

        # All vehicles are currently available
        vehicles["avail_t"] = self.time
        vehicles[["avail_x", "avail_y"]] = vehicles[["x", "y"]]

        # The next time at which vehicles will trigger an epoch: never
        vehicles["epoch_t"] = self._NEVER

        # Done. Set to the env's vehicles
        self._vehicles = vehicles


    def get_noop_action(self, ):
        """Returns the do-nothing action."""

        noop = {
            # Don't reject any of the pending requests
            "req_rejections": np.full((self.num_pending_requests,), fill_value=0, dtype=np.int),
            # Don't assign any vehicles to any of the pending requests
            "req_assgts": np.full((self.num_pending_requests,), fill_value=self._V, dtype=np.int),
            # Don't give any vehicles a repositioning assignment
            "reposition": np.full((self._V,), fill_value=self._D, dtype=np.int)
        }
        
        return noop


    def get_random_action(self, ):
        """Returns a random action."""

        # Initialize our random action
        random_action = self.action_space.sample()
        
        # If a vehicle shows up more than once in the "req_assgts" array, we need
        # to replace its other values with self._V to indicate no assignment.
        # One way to do that is to i) note the unique vehicles assigned to the requests,
        # ii) keep those as they are, and iii) replace everything else with self._V.
        # Step (i)
        vals, indices = np.unique(random_action["req_assgts"], return_index=True)
        # Step (iii)
        random_action["req_assgts"] = np.full_like(random_action["req_assgts"], fill_value=self._V)
        # Step (ii)
        random_action["req_assgts"][indices] = vals

        # Ensure that any vehicle requiring an assignment gets one.
        needs_fixing = (self._vehicles["j1m"] == Jobs.NULL).to_numpy() & (random_action["reposition"] == self._D)
        # Better ways to do this; for now, just send these vehicles to the first lot
        random_action["reposition"][needs_fixing] = 0

        return random_action
    
    
    def _generate_requests(self) -> None:
        """Sets the environment's requests for an episode."""

        # First, sample a day of the week in which the episode is to take place
        dow = self._request_sampler.choice(self._dow_wts.index, p=self._dow_wts.values)
        self._dow = dow

        # Then sample requests for that day of the week
        requests = self._trips_df.loc[self._trips_df.dow == dow, :].sample(
            n=self._num_requests,
            replace=True,
            random_state=self._request_sampler.bit_generator,
            weights=self._trips_df["n_trips"]
        )

        # Adjust the requests' starting window according to the index we
        # denote as episodes' beginning
        requests["t_15min"] = (requests["t_15min"] - self._EPS_START_T_WINDOW) % self._NUM_T_WINDOW

        # Set requests times
        requests["time"] = (
            15 * 60 * requests["t_15min"]
            + self._request_sampler.uniform(size=self._num_requests) * 15 * 60
        )
        requests = requests.drop(["dow", "t_15min", "n_trips"], axis=1)

        # Make sure that all request times are valid
        assert ((0 < requests["time"]) & (requests["time"] < self._MAX_TIME)).all(), "There are requests with invalid release times."

        # Sample an origin and destination pt in the zones
        requests[['ox', 'oy']] = np.vstack(requests['puzone'].apply(self._geom.rand_pt_in_zone).to_numpy())
        requests[['dx', 'dy']] = np.vstack(requests['dozone'].apply(self._geom.rand_pt_in_zone).to_numpy())
        
        # Get the requests' processing time (travel time)
        requests['pt'] = self._geom.travel_time(
            o=requests[['ox','oy']].to_numpy(),
            d=requests[['dx','dy']].to_numpy(),
            hr=self._get_hr_from_time(requests["time"].to_numpy()),
            pairwise=False,
        )

        # Get the requests' value (revenue)
        requests['value'] = self._get_request_value(
            origin=requests[["ox", "oy"]].to_numpy(),
            destination=requests[["dx", "dy"]].to_numpy()
        )

        # Add a dummy job after the end of the horizon
        requests = pd.concat(
            [
                requests,
                pd.DataFrame(
                    [{
                        "ox": self._NULL_X,
                        "dx": self._NULL_X,
                        "oy": self._NULL_Y,
                        "dy": self._NULL_Y,
                        "time": self._NEVER,
                        "pt": 0.0,
                        "value":0.0,
                    }]
                )],
            axis=0,
            ignore_index=True
        )

        # Sort by time and reset the index (to be a RangeIndex)
        requests = requests.sort_values(by="time").reset_index(drop=True)

        # Mark all as unreleased and unrejected, and set the assigned vehicle to None
        requests["released"] = False
        requests["rejected"] = False
        requests["vehicle"] = None

        # Set the env's current request set
        self._requests = requests

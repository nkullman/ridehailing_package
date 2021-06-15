from enum import IntEnum
from typing import Any, Callable, Tuple, Union
import json
import logging
import pkg_resources

from matplotlib.collections import PolyCollection
from scipy.spatial import distance, Delaunay
import numpy as np
import pandas as pd


class Jobs(IntEnum):
    """Defines the different job types that vehicles may have."""
    IDLE = 0
    REPOSITION = 1
    SETUP = 2
    PROCESS = 3
    NULL = 4


def rotate_points(p, origin=(0,0), degrees=0) -> np.ndarray:
    """Rotates points p about origin by the specified degrees."""

    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    p = np.atleast_2d(p)
    o = np.atleast_2d(origin)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


class RidehailGeometry():
    """
    Defines a geometry for the RidehailEnv environment.

    The geometries defined by this class consist of an area that is divided into zones
    and contains locations at which vehicles can idle while waiting to
    serve future requests.

    The class also contains location-related functions, like sampling points in the zones
    and determining travel times and distances.
    
    """

    # The angle (in degrees) that Manhattan streets are offset from a pure North/East grid.
    _MANHATTAN_ANGLE = 29

    # TODO to update and make more realistic, do a query from opendata:
    # ultimate result: puzone, dozone, hr, AVG(trip_time)
    # filter to pu/dozones we want, where putime/dotime are valid (all possible sources)
    _MANHATTAN_SPEEDS = {
        "min": 0.0022352, # 5 mph (in km/s)
        # "mean": 0.0036, # value used in original paper
        "mean": 0.011176, # 25 mph in km/s: the most common speed limit in Manhattan
        "max": 0.0134112, # 30 mph (in km/s)
    }


    def __init__(
        self,
        seed: int,
        distances: str = "manhattan",
        stochastic: bool = False,
    ):
        """Initializes a RidehailGeometry object.

        Args:
            seed: Seed for the geometry's randomness
            speed: Vehicles' mean speed sans traffic
            distances: Method to compute distances; must be one of the following:
                "euclidean", "manhattan"
            stochastic: Whether travel times should be stochastic.
        """

        self.seed = seed
        self.stochastic = stochastic
        self.distances = distances

        self.rotated = self.distances == "manhattan"
        self._distance_func = self._get_distance_function(distances)

        # Load the zones file and populate related class parameters
        self._load_zones(rotate=self.rotated)

        # Load the lots file and populate related class parameters
        self._load_lots(rotate=self.rotated)

        # Load speed-related data
        self._load_speeds()

        # Perform whatever initial seeding needs to be done
        self._initial_seeding()

    
    def _initial_seeding(self):

        # We're going to have 2 distinct generators:
        # One for sampling trip times (actually, trip speeds).
        # And one for sampling points within zones.

        # Create a seed spawner. This will be called twice for each episode:
        # Once for each of our samplers.

        self._seed_spawner = np.random.SeedSequence(self.seed)

        # Initialize our generators to None. They'll be properly initialized
        # when the environment is reset.
        self.trip_time_sampler = None
        self.pt_in_zone_sampler = None
    
    
    def reseed(self):
        """Reseeds all RNGs"""

        # Use our generator to create two new seeds.
        spawns = self._seed_spawner.spawn(2)

        # Use them to seed our vehicle and request sampler for the episode.
        self.trip_time_sampler = np.random.default_rng(spawns[0])
        self.pt_in_zone_sampler = np.random.default_rng(spawns[1])

    
    def _get_distance_function(self, distances: str) -> Callable[..., Any]:
        
        if "euclidean" in distances:
            logging.info("Using Euclidean distances.")
            return self._dist_euclidean

        elif "manhattan" in distances:
            logging.info("Using Manhattan distances.")
            return self._dist_manhattan

        else:
            raise ValueError(f"Unsupported distances type: {distances}")
    
    
    def _load_zones(self, rotate: bool=False) -> None:

        # Load the zones info from file
        stream = pkg_resources.resource_stream(__name__, 'data/taxizones.geojson')
        data = json.load(stream)['features']
        df_data = [
            {
                "zone_id": int(d["properties"]["zone_id"]) ,
                "zone_name": d["properties"]["zone_name"],
                "coordinates": np.array(d["geometry"]["coordinates"])
            }
            for d in data
        ]
        self.zones = pd.DataFrame(df_data)
        # Make sure that all of our zone polygons are single entities
        bad_polys = (self.zones['coordinates'].map(np.ndim) == 3) & (self.zones['coordinates'].map(len).max() > 1)
        if bad_polys.any():
            raise ValueError("Zone polygons must be single entities")

        # Now that we know we don't need it, get rid of zones' first dimension
        self.zones['coordinates'] = self.zones['coordinates'].apply(np.reshape, args=((-1,2),))

        # Ensure proper dtype of the IDs
        self.zones['zone_id'] = self.zones['zone_id'].astype(int)
        
        # Set the ranges from the zones.
        self._xrange, self._yrange = self._compute_ranges(self.zones)

        # Note the origin about which we will perform our rotations, if desired.
        # This is the center of the bounding box for our zones.
        self._rotation_origin = (
            self._xrange[0] + np.ptp(self._xrange)/2.0,
            self._yrange[0] + np.ptp(self._yrange)/2.0
        )

        # Rotate the coordinates of the zone, if desired
        if rotate:
            self.zones['coordinates'] = self.zones['coordinates'].apply(rotate_points, args=(self._rotation_origin, self._MANHATTAN_ANGLE))
            # Update the ranges based on the rotated zones.
            self._xrange, self._yrange = self._compute_ranges(self.zones)
        
        # Add triangulation to the zones
        self.zones = self._triangulate_zones(self.zones)

        # Add on the centroids
        self.zones = self._add_centroids(self.zones)

        # Set the number of zones
        self.num_zones = len(self.zones)

        # Set the zones' locations as a complex array
        self.zone_locs = np.empty((1, self.num_zones), dtype=complex)
        self.zone_locs.real = np.vstack(self.zones["centroids"].values)[:, 0]
        self.zone_locs.imag = np.vstack(self.zones["centroids"].values)[:, 1]

        # Get a mapping from real zone IDs to local zone IDs
        self.zone_id_map = pd.Series(data=self.zones["zone_id"].index, index=self.zones["zone_id"].values)

        # And create a collection of polygons representing the zones. This is useful for rendering.
        self.zone_polys = PolyCollection(
            self.zones['coordinates'].tolist(),
            closed=False,
            facecolors='white',
            edgecolors='black'
        )


    def _compute_ranges(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        xs = df['coordinates'].map(lambda poly_coords: poly_coords[:, 0])
        ys = df['coordinates'].map(lambda poly_coords: poly_coords[:, 1])
        return (
            np.array([xs.map(min).min(), xs.map(max).max()]),
            np.array([ys.map(min).min(), ys.map(max).max()])
        )

    
    def _triangulate_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds triangulation-related columns to df."""
        
        # Map each taxi zone to an (n, 2) array, where n is the number of points that
        # define the taxi zone. Note that we trim off the last pt from each coordinate set since it
        # is a duplicate of the first.
        pts = df['coordinates'].map(lambda coords: coords[:-1,:])
        # Triangulate each of those point sets, then get the vertices of the triangles
        df['tris'] = pts.map(Delaunay).map(lambda tri: tri.points[tri.simplices])

        # Define a function that computes triangles' areas
        def _tri_area(tris):
            """See https://en.wikipedia.org/wiki/Shoelace_formula"""
            assert tris.ndim == 3 and tris.shape[-2:] == (3,2)
            xs = tris[..., 0]
            ys = tris[..., 1]
            return 0.5 * np.abs(
                np.sum(xs * np.roll(ys, 1, axis=-1), axis=-1)
                - np.sum(ys * np.roll(xs, 1, axis=-1), axis=-1)
            )
    
        # Use it to compute the area of the triangles that comprise the zones
        df['tri_areas'] = df['tris'].map(_tri_area)
        
        # And lastly, determine triangles' proportion of the total zone area
        df['tri_rel_areas'] = df['tri_areas'].map(lambda areas: areas / areas.sum())

        return df


    def _add_centroids(self, df: pd.DataFrame) -> pd.Series:
        """Adds a column to the dataframe with the zones' centroids."""

        # Get the center point of each triangle that comprises each zone
        df['tri_means'] = df['tris'].map(lambda tris_pts: np.mean(tris_pts, axis=1))

        # Get the center point of the zone by taking the average of its triangles'
        # center points, weighted by the triangles' areas.
        df['centroids'] = (
            df[['tri_means', 'tri_rel_areas']]
            .apply(
                lambda row:
                np.average(row['tri_means'], axis=0, weights=row['tri_rel_areas']),
                axis=1
            )
        )

        return df
    
    
    def _load_lots(self, rotate: bool=False) -> None:

        stream = pkg_resources.resource_stream(__name__, 'data/lots.csv')
        self.lots = pd.read_csv(stream, index_col="id")

        # Set the number of lots
        self.num_lots = len(self.lots)

        # Rotate the lots if necessary
        if rotate:
            self.lots[['x', 'y']] = rotate_points(
                self.lots[['x', 'y']].to_numpy(),
                origin=self._rotation_origin,
                degrees=self._MANHATTAN_ANGLE
            )

        # Set the lots' locations as a complex array
        self.lot_locs = np.empty((1, self.num_lots), dtype=complex)
        self.lot_locs.real = self.lots["x"].values
        self.lot_locs.imag = self.lots["y"].values


    def _load_speeds(self) -> None:

        # Load the speeds data from file
        try:
            stream = pkg_resources.resource_stream(__name__, 'data/speeds.csv')
            self.speeds = pd.read_csv(stream)
        except:
            logging.warning("Could not read speeds data from file. Using default speeds.")
            self.speeds = self._create_default_speeds_df()

        self._check_speeds()

    
    def _check_speeds(self) -> None:
        """Some simple checks on our speeds data."""

        # If the user requested a deterministic environment, then we force the standard
        # deviation on our speeds data to be 0.
        if not self.stochastic:
            # If the user provided non-zero speeds, then warn about this change
            if any(self.speeds["speed_stddev"] > 0):
                logging.warning(
                    "Speeds data implies stochastic travel times (nonzero standard deviations), "
                    "but non-stochastic environment requested. Using non-stochastic environment "
                    "and setting all standard deviations to zero."
                )
            self.speeds["speed_stddev"] = 0

        # If the user requested a stochastic environment but all standard deviations are 0,
        # we warn them about this (but do nothing).
        if self.stochastic and all(self.speeds["speed_stddev"] == 0):
            logging.warning(
                "Stochastic environment requested, but speeds data implies non-stochastic "
                "(all standard deviations are 0)."
            )

        
    def _create_default_speeds_df(self) -> pd.DataFrame:

        HRS_PER_DAY = 24
        
        # Start by creating a square dataframe that specifies a single mean speed for each pair of zones
        speeds = pd.DataFrame(index=self.zones.index, columns=self.zones.index, data=self._MANHATTAN_SPEEDS["mean"])

        # Melt the df to get a column for origin zone and destination zone
        speeds = pd.melt(
            # Start by moving the origin zone into the df
            speeds.reset_index().rename(columns={"index": "zone_o"}),
            # Keep that column fixed
            id_vars="zone_o",
            # Unpivot the rest of the columns, their names now corresponding to the destination zone.
            var_name="zone_d",
            # The values are the mean speed
            value_name="speed_mean"
        )

        # Count the number of connections between zones
        n_zn_cnxns = len(speeds)

        # Speed data is assumed to be given hourly
        speeds = pd.concat([speeds for hr in range(HRS_PER_DAY)], axis=0, ignore_index=True)

        # Add on the 'hr' column
        speeds["hr"] = np.arange(HRS_PER_DAY).repeat(n_zn_cnxns)

        # Add on a column for the standard deviation. 0 by default
        speeds["speed_stddev"] = 0

        return speeds


    @property
    def x_range(self):
        return self._xrange


    @property
    def y_range(self):
        return self._yrange


    def add_init_weights_to_lots(self, trips: pd.DataFrame, eps_init_hr: int) -> None:
        """Add init_weights column to self.lots indicating probability of vehicle spawning."""

        # Given episodes' first hour, compute their last hour
        eps_final_hr = (eps_init_hr - 1) % 24

        # Count the number of dropoffs in each zone in the final hour (zone, pct_drops)
        dropoffs_by_zone = (
            trips.loc[trips['hr'] == eps_final_hr, :]
            .groupby('dozone')
            .size()
            .rename("pct_dropoffs")
            .rename_axis("zone")
        )
        dropoffs_by_zone = dropoffs_by_zone / dropoffs_by_zone.sum() # get pct of dropoffs in each zone

        # Get number of lots per zone (zone, num_lots)
        lot_zone_ct = (
            self.lots
            .groupby('zone')
            .size()
            .rename("num_lots")
            .rename_axis("zone")
        )

        # Get the pct wt to be given to a lot in each of the zones
        wts_df = pd.concat([dropoffs_by_zone, lot_zone_ct], axis=1)
        wts_df['init_weight'] = (wts_df['pct_dropoffs'] / wts_df['num_lots'])

        # Map these values onto our lots
        self.lots = self.lots.merge(wts_df.reset_index()[["zone", "init_weight"]], on="zone", how="left")
        
        # Replace any missing weights (zones without dropoffs or zones without lots)
        # or inf weights (zones with dropoffs but no CSs) with 0s.
        self.lots["init_weight"] = self.lots["init_weight"].replace(np.inf, np.nan).fillna(0)
        
        # Normalize to ensure the sum of the weights is one
        self.lots["init_weight"] = self.lots["init_weight"] / self.lots["init_weight"].sum()

        return


    def dist(self, o: np.ndarray, d: np.ndarray, pairwise: bool=False) -> Union[np.ndarray, float]:
        """Compute the distance between origin(s) `o` and destination(s) `d`.

        If o and d are each a single pair of coordinates, then returns a scalar.
        
        If o and d have different shapes, then returns the distance matrix
        whose shape is (len(o), len(d)).

        If o and d have the same shape, then returns the element-wise distance
        between members in o and d.

        If o and d have the same shape and you want the distance matrix,
        then set pairwise=True.

        """

        return self._distance_func(o, d, pairwise)

        
    def _dist_euclidean(self, o: np.ndarray, d: np.ndarray, pairwise: bool) -> np.ndarray:
        """Compute the euclidean distance between origin(s) `o` and destination(s) `d`.

        If o and d are each a single pair of coordinates, then returns a scalar.
        
        If o and d have different shapes, then returns the distance matrix
        whose shape is (len(o), len(d)).

        If o and d have the same shape, then returns the element-wise distance
        between members in o and d.

        If o and d have the same shape and you want the distance matrix,
        then set pairwise=True.

        """
        
        # Just want distance between two (non-complex) points
        if o.shape == d.shape == (2,):
            # Just get the difference between the complex points
            return np.abs(np.complex(*o)-np.complex(*d))
        
        # Want the distance matrix
        elif o.shape != d.shape or pairwise:

            # Transpose the first to leverage broadcasting
            result = np.abs(self._to_complex_array(o).T - self._to_complex_array(d))
            
            # Return vector if result is 1xN or Nx1
            return result.flatten() if 1 in result.shape else result

        # Want element-wise distances
        else:

            result = np.abs(self._to_complex_array(o) - self._to_complex_array(d)).flatten()

            # If length is one, return scalar (only happens if passed two complex singletons; unlikely but possible)
            return result[0] if result.size == 1 else result


    def _to_complex_array(self, coords: np.ndarray) -> np.ndarray:
        """Input coords should have shape (2,) or (N,2) (2 for x,y)."""

        # See if it's already a properly formatted complex array
        if coords.dtype == complex and coords.shape == (1, coords.size):
            return coords

        # Assert proper formatting for conversion
        assert coords.shape == (coords.size / 2, 2) or coords.shape == (2,), (
            f"Improper dimensions for conversion to complex. Shape: {coords.shape} (should be (N,2))."
        )

        # Make complex array
        result = np.empty((1, coords[..., 0].size), dtype=complex)
        result.real = coords[..., 0]
        result.imag = coords[..., 1]
        
        return result


    def xy_to_zone(self, xys: np.ndarray) -> np.ndarray:
        """Converts an array of XY coordinates into an array of taxi zone IDs.
        
        Coordinate pairs are assigned to the zone whose centroid they are closest to.
        
        """

        # Get the distance from xys to zone centroids
        dists = self._dist_euclidean(xys, self.zone_locs, pairwise=True)
        
        # Return the closest one
        return np.argmin(dists, axis=-1)


    def _dist_manhattan(self, o: np.ndarray, d: np.ndarray, pairwise: bool) -> np.ndarray:
        """Compute the Manhattan distance between origin(s) `o` and destination(s) `d`.

        If o and d are each a single pair of coordinates, then returns a scalar.
        
        If o and d have different shapes, then returns the distance matrix
        whose shape is (len(o), len(d)).

        If o and d have the same shape, then returns the element-wise distance
        between members in o and d.

        If o and d have the same shape and you want the distance matrix,
        then set pairwise=True.

        """
        
        # Just want distance between two points
        if o.shape == d.shape == (2,):
            # Just get the difference between the complex points
            return np.abs(o-d).sum()
        
        # Want the distance matrix
        elif o.shape != d.shape or pairwise:

            # Make sure each is 2D
            o = o.reshape(-1, 2)
            d = d.reshape(-1, 2)

            # Use SciPy for the distance matrix
            result = distance.cdist(o, d, "cityblock")
            
            # Return vector if result is 1xN or Nx1
            return result.flatten() if 1 in result.shape else result

        # Want element-wise distances
        else:

            result = np.abs(o-d).sum(axis=1)
            return result[0] if result.size == 1 else result


    def rand_pt_in_zone(self, zone_ids: int) -> np.ndarray:

        # Get the index of the zone in the zones df
        zone_id_local = self.zone_id_map[zone_ids]

        # Sample a triangle from its triangles
        tri = self.zones.loc[zone_id_local, "tris"][self.pt_in_zone_sampler.choice(
            a=self.zones.loc[zone_id_local, "tris"].shape[0],
            p=self.zones.loc[zone_id_local, "tri_rel_areas"]
        )]
        
        # Sample a point inside the sampled triangle
        r1, r2 = tuple(self.pt_in_zone_sampler.uniform(size=2))
        sqrt_r1 = np.sqrt(r1)
        return np.array((1 - sqrt_r1) * tri[0] + (sqrt_r1 * (1 - r2)) * tri[1] + (r2 * sqrt_r1) * tri[2])

        
        # TODO vectorize this function's sampling so that `zone_ids` could be an np.ndarray of ints
        #
        # Then do this kinda thing where we sample from each row simultaneously.
        # Need a matrix ("tris") that has a row for each zone, a column for each triangle in the zone.
        # All rows need to be equal length, where the length is the max num of triangles for any zone.
        # Values are the tri_rel_areas, right-padded with nans wgere rows (zones) have fewer triangles.
        # Then do like below to sample a triangle sample from each zone.
        #
        # sampled_tris = (tris.cumsum(1) > self.pt_in_zone_sampler.uniform(self.num_zones)[:,None]).argmax(1)
        #
        # Then something like...
        # 
        #     n = len(zone_id)
        #     sqrt_r1 = np.sqrt(self.pt_in_zone_sampler.uniform(self.num_zones))
        #     r2 = self.pt_in_zone_sampler.uniform(self.num_zones)
        #     return np.array(
        #         (1-sqrt_r1)*tris[:,0] + (sqrt_r1*(1-r2))*tris[:,1] + (r2*sqrt_r1)*tris[:,2]
        #     )
    
    
    def travel_time(self, o: np.ndarray, d: np.ndarray, hr: Union[int, np.ndarray]=9, mean: bool=False, pairwise: bool=False) -> np.ndarray:
        """Computes travel times between origin(s) o and destinations (d).

        Assumes that travel is happening in hour hr (defaults to 9am).

        """

        if pairwise and not isinstance(hr, (int, np.integer)):
            raise ValueError("Pairwise travel times can only be computed for a single time.")
        
        if isinstance(hr, np.ndarray) and (hr.shape != (len(o),) or hr.shape != (len(d),)):
            # Not doing pairwise. If hr is an array, then it should be the same length
            # as o and d.
            raise ValueError("hr, o, and d should have the same length.")

        # The distances between origins o and destinations d
        dists = self.dist(o, d, pairwise=pairwise)
        dists_shape = np.shape(dists)

        # The zones of the origins and destinations
        zones_o = self.xy_to_zone(o)
        zones_d = self.xy_to_zone(d)
        
        # If we have a matrix or if we have a 1xN or Nx1 situation, then we need to do some reshaping...
        if np.ndim(dists) == 2:
            dists = dists.flatten()
            zones_o = np.repeat(zones_o, len(d))
            zones_d = np.tile(zones_d, len(o))

        elif np.ndim(dists) == 1 and o.shape != d.shape:
            # Dists is fine, but we need to reshape one of our zones arrays.
            if zones_o.size > 1:
                # We have more than one origin. Need to repeat our single destination for each one of these.
                zones_d = np.repeat(zones_d, len(o))
            elif zones_d.size > 1:
                # We have more than one destination. Need to repeat our single origin for each one of these.
                zones_o = np.repeat(zones_o, len(d))
            else:
                raise ValueError("Bad dimensions")
        
        # Make sure at least one element is an array (for pandas' sake)
        if np.ndim(dists) == 0:
            dists = np.asarray(dists).reshape(-1)

        # Create a df for the required travel time computations.
        df = pd.DataFrame(data={"zone_o": zones_o, "zone_d": zones_d, "hr": hr, "dist": dists})
        
        # Join on the speed data
        df = df.merge(self.speeds, how="left", on=["zone_o", "zone_d", "hr"])

        # Get the realized speeds
        if mean:
            # The realized speed is just the mean speed
            df["speed_realized"] = df["speed_mean"]
        
        else:
            # Sample a speed for each trip, and clip it to be within our min and max
            df["speed_realized"] = (
                np.clip(
                    self.trip_time_sampler.normal(loc=df["speed_mean"], scale=df["speed_stddev"]),
                    self._MANHATTAN_SPEEDS["min"],
                    self._MANHATTAN_SPEEDS["max"]
                )
            )

        # Get the travel times
        df["travel_time"] = df["dist"] / df["speed_realized"]

        # Return the values, appropriately shaped
        return df["travel_time"].values.reshape(dists_shape)

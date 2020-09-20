from pyspark.sql import DataFrame
from functools import reduce
import math
from pyspark.sql import functions as F
import pygeohash as gh


def union_tables(tables: list):
    return reduce(DataFrame.union, tables)


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371  # km
    if (lat1 is None) | (lon1 is None) | (lat2 is None) | (lon2 is None):
        return None
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


def calculate_bearing_degrees(latitude_1, longitude_1, latitude_2, longitude_2):
    diff_longitude = F.radians(longitude_2 - longitude_1)

    r_latitude_1 = F.radians(latitude_1)
    r_longitude_1 = F.radians(longitude_1)
    r_latitude_2 = F.radians(latitude_2)
    r_longitude_2 = F.radians(longitude_2)

    y = F.sin(diff_longitude) * F.cos(r_longitude_2)
    x = (
            F.cos(r_latitude_1) * F.sin(r_latitude_2) -
            F.sin(r_latitude_1) * F.cos(r_latitude_2) * F.cos(diff_longitude)
    )

    return F.degrees(F.atan2(x, y))


def percentile(percentile, column):
    return F.expr(f'percentile_approx({column}, {percentile})')


def geocode_latitude_and_longitude(latitude, longitude, precision):
    return gh.encode(
        latitude=latitude,
        longitude=longitude,
        precision=precision
    )

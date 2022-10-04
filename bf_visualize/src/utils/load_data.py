"""
This file contains all the functions related to the data loading
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
from obspy.geodetics.base import locations2degrees
from pyasdf import ASDFDataSet
from tqdm import tqdm


@dataclass
class DataInfo:
    """Data class recording csv file info
    """
    event_id: str
    station: str
    origin_time: UTCDateTime
    ptime: UTCDateTime
    stime: UTCDateTime
    pstime: UTCDateTime
    dist: float


def load_data_info(path: str) -> Dict[Tuple[str, str], DataInfo]:
    """Load arrival time csv file

    Args:
        path (str): the arrival data csv path

    Returns:
        Dict[str,float]: the dict containing the required information
    """
    df = pd.read_csv(path)
    res = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        # keys
        event_id = row["EVENT_ID"]
        station = row["STATION"]
        # we record the time as the absolute time
        origin_time = UTCDateTime(row["ORIGIN_TIME"])
        ptime = origin_time + \
            row["PTIME"] if (not np.isnan(row["PTIME"])) else None
        stime = origin_time + \
            row["STIME"] if (not np.isnan(row["STIME"])) else None
        pstime = origin_time + \
            row["PSTIME"] if (not np.isnan(row["PSTIME"])) else None
        # add dist
        dist = locations2degrees(
            row["ELAT"], row["ELON"], row["SLAT"], row["SLON"])
        res[(event_id, station)] = DataInfo(
            event_id=event_id,
            station=station,
            origin_time=origin_time,
            ptime=ptime,
            stime=stime,
            pstime=pstime,
            dist=dist
        )
    return res


def load_waveforms(path: str, keys: List[Tuple[str, str]]) -> Dict[str, Stream]:
    """Load the waveform from the ASDF file

    Args:
        path (str): the asdf path
        keys (List[Tuple[str,str]]): the keys as a list with (event_id,station)

    Returns:
        Dict[str,Stream]: the stream for each (event_id,station) pair
    """
    res = {}
    with ASDFDataSet(path, mode="r") as ds:
        for event_id, station in tqdm(keys, desc="loading waveforms"):
            asdf_key = f"{event_id}.{station}"
            res[(event_id, station)] = ds.waveforms[asdf_key].raw_recording
    return res


def load_ps_info(path: str):
    data = np.loadtxt(path, dtype=str)
    res = {}
    for row in data:
        event_id, station = row[0].split(".")
        res[(event_id, station)] = {
            "s": float(row[1]),
            "e": float(row[2]),
            "snr": float(row[3]),
            "index": int(row[4])
        }
    return res


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    load_data_info(os.path.join(
        dir_path, "../../data/arrival_info_by_ziyi_20220921_add_sta_info.csv"))

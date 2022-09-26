""" 
Process the waveform
"""
from typing import Dict, Tuple

from obspy import UTCDateTime, Stream

from .load_data import DataInfo


def parepare_reference_time(ref_type: str, data_info: Dict[Tuple[str, str], DataInfo]) -> Dict[Tuple[str, str], UTCDateTime]:
    """Prepare the reference time for further data slicing

    Args:
        ref_type (str): the reference time type
        data_info (Dict[Tuple[str,str],DataInfo]): the data info

    Returns:
        Dict[Tuple[str,str],UTCDateTime]: the dict recording the reference time
    """
    res = {}
    if ref_type == "P":
        for key in data_info:
            res[key] = data_info[key].ptime
    elif ref_type == "PS":
        for key in data_info:
            res[key] = data_info[key].pstime
    elif ref_type == "S":
        for key in data_info:
            res[key] = data_info[key].stime
    else:
        raise Exception("ref type must be P, S, or PS.")
    return res


def slice_waveform(waves: Dict[str, Stream], refs: Dict[Tuple[str, str], UTCDateTime], left_win: float = 10., right_win: float = 50.) -> Dict[str, Stream]:
    """Slice the wave streams based on ref time

    Args:
        waves (Dict[str,Stream]): the raw waveform
        refs (Dict[Tuple[str,str],UTCDateTime]): the reference times
        left_win (float, optional): the window length left to ref time. Defaults to -10..
        right_win (float, optional): the window length right to ref time. Defaults to 50..

    Returns:
        Dict[str,Stream]: the sliced waveform
    """
    res = {}
    for key in waves:
        stream = waves[key].copy()
        ref = refs[key]
        if stream[0].stats.starttime > ref-left_win or stream[0].stats.endtime < ref+right_win:
            raise Exception(
                f"{key} has range {stream[0].stats.starttime}->{stream[0].stats.endtime}, but asked {ref-left_win}->{ref+right_win}.")
        res[key] = stream.slice(starttime=ref-left_win, endtime=ref+right_win)
    return res


def post_process(waves: Dict[str, Stream], freqmin: float, freqmax: float, sampling_rate: int = 100) -> Dict[str, Stream]:
    """Post process the waveforms

    Args:
        waves (Dict[str, Stream]): the sliced waveform
        freqmin (float): the freq min for filtering
        freqmax (float): the freq max for filtering
        sampling_rate (int, optional): the sampling rate to resample. Defaults to 100.

    Returns:
        Dict[str, Stream]: the post processed waveforms
    """
    res = {}
    for key in waves:
        stream = waves[key].copy()
        stream.detrend("demean")
        stream.detrend("linear")
        stream.taper(max_percentage=0.05, type="hann")
        stream.interpolate(sampling_rate=sampling_rate)
        stream.filter("bandpass", freqmin=freqmin,
                      freqmax=freqmax, corners=2, zerophase=True)
        res[key] = stream
    return res

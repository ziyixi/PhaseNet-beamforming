""" 
Process the waveform
"""
from typing import Dict, Tuple

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from tqdm import tqdm

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


def slice_waveform(waves: Dict[Tuple[str, str], Stream], refs: Dict[Tuple[str, str], UTCDateTime], left_win: float = 10., right_win: float = 50.) -> Dict[Tuple[str, str], Stream]:
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
    for key in tqdm(waves, desc="slicing"):
        stream = waves[key].copy()
        ref = refs[key]
        if ref is not None:
            if stream[0].stats.starttime > ref-left_win or stream[0].stats.endtime < ref+right_win:
                raise Exception(
                    f"{key} has range {stream[0].stats.starttime}->{stream[0].stats.endtime}, but asked {ref-left_win}->{ref+right_win}.")
            res[key] = stream.slice(
                starttime=ref-left_win, endtime=ref+right_win)
    return res


def post_process(waves: Dict[Tuple[str, str], Stream], freqmin: float, freqmax: float, sampling_rate: int = 100) -> Dict[Tuple[str, str], Stream]:
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
    for key in tqdm(waves, desc="processing"):
        stream = waves[key].copy()
        stream.detrend("demean")
        stream.detrend("linear")
        stream.taper(max_percentage=0.05, type="hann")
        stream.interpolate(sampling_rate=sampling_rate)
        stream.filter("bandpass", freqmin=freqmin,
                      freqmax=freqmax, corners=2, zerophase=True)
        res[key] = stream
    return res


def select_based_on_snr(waves: Dict[Tuple[str, str], Stream], noise_win: Tuple[float, float], signal_win: Tuple[float, float], min_snr: float = 3) -> Dict[Tuple[str, str], Trace]:
    """Select the max SNR trace in streams

    Args:
        waves (Dict[str, Stream]): the waveforms
        noise_win (Tuple[float, float]): the noise win range, measured from the trace start time
        signal_win (Tuple[float, float]): the signal win range, measured from the trace start time

    Returns:
        Dict[str, Trace]: the processed waveforms
    """
    noise_start, noise_end = noise_win
    signal_start, signal_end = signal_win
    res = {}
    for key in waves:
        stream = waves[key]
        maxsnr, maxtrace = 0, None
        for trace in stream:
            start_time = trace.stats.starttime
            noise = trace.slice(start_time+noise_start,
                                start_time+noise_end).data
            signal = trace.slice(start_time+signal_start,
                                 start_time+signal_end).data
            noise_power = np.mean(noise**2)
            signal_power = np.mean(signal**2)
            snr = signal_power/noise_power
            if snr > maxsnr:
                maxsnr = snr
                maxtrace = trace
        if maxtrace is not None:
            # do normalize
            maxtrace.normalize()
            if maxsnr < min_snr:
                res[key] = None
            else:
                res[key] = maxtrace
        else:
            raise Exception(f"calculation of {key} leads to snr==0")
    return res


def post_process_ps(waves: Dict[Tuple[str, str], Stream], ps_info, sampling_rate: int = 100) -> Dict[Tuple[str, str], Stream]:
    res = {}
    for key in tqdm(ps_info, desc="processing"):
        stream = waves[key].copy()
        stream.detrend("demean")
        stream.detrend("linear")
        stream.taper(max_percentage=0.05, type="hann")
        stream.interpolate(sampling_rate=sampling_rate)
        stream.filter("bandpass", freqmin=ps_info[key]["s"],
                      freqmax=ps_info[key]["e"], corners=2, zerophase=True)
        res[key] = stream
    return res


def select_based_on_snr_ps(waves: Dict[Tuple[str, str], Stream], ps_info, min_snr: float = 3) -> Dict[Tuple[str, str], Trace]:
    res = {}
    for key in ps_info:
        stream = waves[key]
        if ps_info[key]["snr"] < min_snr:
            continue
        tr = stream[ps_info[key]["index"]]
        tr.normalize()
        res[key] = tr
    return res

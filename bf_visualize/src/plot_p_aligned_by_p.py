""" 
Plot the P wave aligned P arrival seismic array, one pdf per station.
"""
import os
import pickle
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pygmt
from loguru import logger
from obspy import Trace
from tqdm import tqdm

from .utils.load_data import load_data_info, load_waveforms
from .utils.process_waveform import (DataInfo, parepare_reference_time,
                                     post_process, select_based_on_snr,
                                     slice_waveform)

dirpath = tempfile.mkdtemp()

dir_path = os.path.dirname(os.path.realpath(__file__))

# * configs
CACHE_PATH = os.path.join(dir_path, "../cache/plot_p_aligned_by_p.pkl")
DATA_INFO_CSV_PATH = os.path.join(
    dir_path, "../data/arrival_info_by_ziyi_20220921_add_sta_info.csv")
ASDF_PATH = os.path.join(dir_path, "../data/tongaml.h5")
LEFT_WIN = 20
RIGHT_WIN = 40
FREQMIN = 1
FREQMAX = 8
SAMPLING_RATE = 100
NOISE_WIN = (0, 15)
SIGNAL_WIN = (15, 25)
# * plotting configs
SCALE = 0.5
MIN_SNR = 10


def show_array(station: str, info_this_station: List[DataInfo], waveforms: Dict[Tuple[str, str], Trace]) -> None:
    # * the first step is to prepare the plotting data
    toplot_x = OrderedDict()
    toplot_y = OrderedDict()

    count = 1
    for data_info in info_this_station:
        key = (data_info.event_id, data_info.station)
        trace = waveforms[key]
        # trace==None when SNR is small
        if trace is not None:
            toplot_x[data_info.event_id] = np.linspace(
                -LEFT_WIN, RIGHT_WIN, len(trace.data))
            toplot_y[data_info.event_id] = trace.data*SCALE+count+1
            count += 1
    # * plotting
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL="12p", MAP_LABEL_OFFSET="6p",
                 FONT_ANNOT_PRIMARY="12p", MAP_FRAME_TYPE="plain")
    fig.basemap(projection="X8i/18i", region=[-LEFT_WIN, RIGHT_WIN, 0, len(toplot_x)+2],
                frame=["WSen", "xaf+lTime(s)", "yaf+lIndex"])

    for key in tqdm(toplot_x, desc=f"plotting {station}"):
        fig.plot(x=toplot_x[key], y=toplot_y[key], pen="0.4p,black")

    tmp_dir = tempfile.mkdtemp()
    save_path = os.path.join(tmp_dir, f"{station}.pdf")
    fig.savefig(save_path)
    print(save_path)


def main():
    # * all the processing part
    if not os.path.isfile(CACHE_PATH):
        logger.info("no cache, loading data info and waveforms.")
        data_infos = load_data_info(DATA_INFO_CSV_PATH)
        raw_waveforms = load_waveforms(ASDF_PATH, list(data_infos.keys()))
        reference_times = parepare_reference_time("P", data_infos)
        # sliced_waveforms might remove some traces without the ref time for certain phases
        sliced_waveforms = slice_waveform(
            raw_waveforms, reference_times, LEFT_WIN, RIGHT_WIN)
        waveforms_with_stream = post_process(sliced_waveforms, FREQMIN,
                                             FREQMAX, sampling_rate=SAMPLING_RATE)
        waveforms = select_based_on_snr(
            waveforms_with_stream, NOISE_WIN, SIGNAL_WIN, min_snr=MIN_SNR)
        with open(CACHE_PATH, 'wb') as handle:
            pickle.dump({
                "data_infos": data_infos,
                "waveforms": waveforms
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logger.info("find cache, loading cache.")
        with open(CACHE_PATH, 'rb') as handle:
            cache = pickle.load(handle)
        data_infos, waveforms = cache["data_infos"], cache["waveforms"]

    # * now we group data_info based on their station name, ranked by their dist
    infos_group_by_station = defaultdict(list)
    for (network, station), data_info in data_infos.items():
        if (network, station) in waveforms:
            infos_group_by_station[station].append(data_info)
    for station in infos_group_by_station:
        infos_group_by_station[station].sort(key=lambda each: each.dist)

    # * now for each key, we plot a pdf indicating its waveforms
    for station in infos_group_by_station:
        if station == "FONI":
            show_array(station, infos_group_by_station[station], waveforms)


if __name__ == "__main__":
    main()

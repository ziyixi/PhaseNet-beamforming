""" 
As we will perform the source side beamforming, it's essencial to look at the events' distribution.
"""
import os

import numpy as np
import pygmt

from .utils.load_data import load_data_info

dir_path = os.path.dirname(os.path.realpath(__file__))
PDF_SAVE_DIR = os.path.join(dir_path, "../pdfs/depth_distribution")
DATA_INFO_CSV_PATH = os.path.join(
    dir_path, "../data/arrival_info_by_ziyi_20220921_add_sta_info.csv")
STATION = "NMKA"


def extract_events_depth():  # pylint: disable=missing-function-docstring
    data_infos = load_data_info(DATA_INFO_CSV_PATH)

    deps = []
    dists = []
    for (_, station), data_info in data_infos.items():
        if station == STATION:
            deps.append(data_info.evdp)
            dists.append(data_info.dist)
    return np.array(deps), np.array(dists)


def main():  # pylint: disable=missing-function-docstring
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL="14p", MAP_LABEL_OFFSET="8p", FONT_ANNOT_PRIMARY="12p",
                 MAP_FRAME_TYPE="plain", MAP_TITLE_OFFSET="8p", FONT_TITLE="14p,black", MAP_FRAME_PEN="1p,black")

    fig.basemap(region=[0, 10, 0, 800], projection="X6i/-6i",
                frame=["WSen", 'xaf+l"Distance (degree)"', 'yaf+l"Depth (km)"'])

    deps, dists = extract_events_depth()
    # print(min(deps), max(deps), min(dists), max(dists))
    fig.plot(x=dists, y=deps, style="c4p", color="red3")

    save_path = os.path.join(PDF_SAVE_DIR, f"{STATION}.pdf")
    fig.savefig(save_path)


if __name__ == "__main__":
    main()

""" 
As we will perform the source side beamforming, it's essencial to look at the events' distribution.
"""
import os

import numpy as np
import pygmt

from .utils.load_data import load_data_info

dir_path = os.path.dirname(os.path.realpath(__file__))
PDF_SAVE_DIR = os.path.join(dir_path, "../pdfs/events_distribution")
DATA_INFO_CSV_PATH = os.path.join(
    dir_path, "../data/arrival_info_by_ziyi_20220921_add_sta_info.csv")
STATION = "NMKA"


def extract_events_locations():  # pylint: disable=missing-function-docstring
    data_infos = load_data_info(DATA_INFO_CSV_PATH)

    lons, lats = [], []
    for (_, station), data_info in data_infos.items():
        if station == STATION:
            lons.append(data_info.evlo)
            lats.append(data_info.evla)
    return np.array(lons), np.array(lats)


def main():  # pylint: disable=missing-function-docstring
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL="14p", MAP_LABEL_OFFSET="8p", FONT_ANNOT_PRIMARY="12p",
                 MAP_FRAME_TYPE="plain", MAP_TITLE_OFFSET="8p", FONT_TITLE="14p,black", MAP_FRAME_PEN="1p,black")

    grd_topo = pygmt.datasets.load_earth_relief(
        resolution="02m", region=[175, 195, -30, -10])

    fig.basemap(region=[175, 195, -30, -10], projection="M6i",
                frame=["WSen", "xaf", "yaf"])
    fig.coast(water="167/194/223")
    fig.grdimage(grd_topo, cmap=os.path.join(
        dir_path, "land_sea.cpt"), shading="+d")

    lons, lats = extract_events_locations()
    # print(max(lons[lons > 0]), min(lons[lons > 0]), max(
    #     lons[lons < 0]), min(lons[lons < 0]), max(lats), min(lats))
    fig.plot(x=lons, y=lats, style="c4p", color="red3")
    # -174.798004,-20.2596
    fig.plot(x=-174.798004, y=-20.2596, style="c12p", color="blue")

    save_path = os.path.join(PDF_SAVE_DIR, f"{STATION}.pdf")
    fig.savefig(save_path)


if __name__ == "__main__":
    main()

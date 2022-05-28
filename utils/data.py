"""Utilities for loading data from publicly available data sets.
"""
import argparse
import glob
import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd


KITTI_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
KITTI_OUT = 'data-cache'


OXTS_COLUMNS = [
    'lat',           # latitude of the oxts-unit (deg)
    'lon',           # longitude of the oxts-unit (deg)
    'alt',           # altitude of the oxts-unit (m)
    'roll',          # roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
    'pitch',         # pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
    'yaw',           # heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
    'vn',            # velocity towards north (m/s)
    've',            # velocity towards east (m/s)
    'vf',            # forward velocity, i.e. parallel to earth-surface (m/s)
    'vl',            # leftward velocity, i.e. parallel to earth-surface (m/s)
    'vu',            # upward velocity, i.e. perpendicular to earth-surface (m/s)
    'ax',            # acceleration in x, i.e. in direction of vehicle front (m/s^2)
    'ay',            # acceleration in y, i.e. in direction of vehicle left (m/s^2)
    'az',            # acceleration in z, i.e. in direction of vehicle top (m/s^2)
    'af',            # forward acceleration (m/s^2)
    'al',            # leftward acceleration (m/s^2)
    'au',            # upward acceleration (m/s^2)
    'wx',            # angular rate around x (rad/s)
    'wy',            # angular rate around y (rad/s)
    'wz',            # angular rate around z (rad/s)
    'wf',            # angular rate around forward axis (rad/s)
    'wl',            # angular rate around leftward axis (rad/s)
    'wu',            # angular rate around upward axis (rad/s)
    'pos_accuracy',  # velocity accuracy (north/east in m)
    'vel_accuracy',  # velocity accuracy (north/east in m/s)
    'navstat',       # navigation status (see navstat_to_string)
    'numsats',       # number of satellites tracked by primary GPS receiver
    'posmode',       # position mode of primary GPS receiver (see gps_mode_to_string)
    'velmode',       # velocity mode of primary GPS receiver (see gps_mode_to_string)
    'orimode']       # orientation mode of primary GPS receiver (see gps_mode_to_string)


def _parse_kitti_oxts(*, path):
    # Load data one entry at a time...
    files = sorted([f for f in glob.glob(os.path.join(path, 'data/*.txt'))])
    data = np.array([np.fromfile(f, sep=' ') for f in files])

    # Corresponding timestamps are in another file...
    with open(os.path.join(path, 'timestamps.txt'), 'r') as f:
        datestr = [l for l in f.readlines()]
    index = pd.DatetimeIndex(data=datestr)
    assert len(index) == data.shape[0]

    # Create data frame ane augment with timestamp seconds column.
    df = pd.DataFrame(data, index=index, columns=OXTS_COLUMNS)
    df['timestamp'] = 1e-9 * index.view(np.int64)
    return df


def get_kitti_data(*, drive, sequence, force_download=False):
    """
    Download raw odometry data from KITTI.

    Note: we download the raw unsynced data since this includes 100Hz OXTS
    outputs.

    Parameters
    ----------
    drive : (str) drive string, e.g., 2011_09_26
    sequence : (int) drive sequence number, e.g., 1
    force_download : (bool) optionally require new download

    Returns
    -------
    df :  pandas dataframe
    """
    os.makedirs(KITTI_OUT, exist_ok=True)
    path = os.path.join(KITTI_OUT, drive)
    if force_download or not os.path.exists(os.path.join(path, f'{drive}_drive_{sequence:04d}_extract')):
        def _download_extract(fin):
            url = f'{KITTI_URL}/{fin}'
            print(f'downloading {url}...')
            # fout, _ = urllib.request.urlretrieve(url)  # downloads to tmp
            fout, _ = urllib.request.urlretrieve(url, filename=f'foo{np.random.randint(1000):04d}')
            print(f'extracting {fout}...')
            with zipfile.ZipFile(fout, 'r') as z:  # extracts to pwd
                z.extractall(path=KITTI_OUT)
        _download_extract(f'{drive}_calib.zip')
        _download_extract(f'{drive}_drive_{sequence:04d}/{drive}_drive_{sequence:04d}_extract.zip')

        # Remove image/velodyne data since we only care about oxts.
        shutil.rmtree(os.path.join(path, f'{drive}_drive_{sequence:04d}_extract', 'image_00'), ignore_errors=True)
        shutil.rmtree(os.path.join(path, f'{drive}_drive_{sequence:04d}_extract', 'image_01'), ignore_errors=True)
        shutil.rmtree(os.path.join(path, f'{drive}_drive_{sequence:04d}_extract', 'image_02'), ignore_errors=True)
        shutil.rmtree(os.path.join(path, f'{drive}_drive_{sequence:04d}_extract', 'image_03'), ignore_errors=True)
        shutil.rmtree(os.path.join(path, f'{drive}_drive_{sequence:04d}_extract', 'velodyne_points'),
                      ignore_errors=True)
    assert os.path.exists(path)

    df = _parse_kitti_oxts(
            path=os.path.join(path, f'{drive}_drive_{sequence:04d}_extract', 'oxts'))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive', required=True, help='drive label e.g., 2011_09_30')
    parser.add_argument('--sequence', required=True, type=int, help='sequence number')
    args = parser.parse_args()

    print(f'downloading drive {args.drive}/sequence {args.sequence}')
    df = get_kitti_data(drive=args.drive, sequence=args.sequence)
    print(df.describe())


if __name__ == '__main__':
    main()

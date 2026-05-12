# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load forage with vertices and ground truth labels for species classification."""
import argparse
import inspect
import json
import os
import laspy

import numpy as np
import open3d as o3d

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i][
                'objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, label_ids, instance_ids, bg_sem=np.array([0])):
    # Filter out background points
    valid_mask = ~np.isin(label_ids, bg_sem)
    mesh_vertices = mesh_vertices[valid_mask]
    instance_ids = instance_ids[valid_mask]
    label_ids = label_ids[valid_mask]

    # Get the number of unique instances
    unique_instance_ids = np.unique(instance_ids)
    num_instances = len(unique_instance_ids)

    # Initialize instance_bboxes
    instance_bboxes = np.zeros((num_instances, 7))

    for i, instance_id in enumerate(unique_instance_ids):
        # Select points corresponding to the current instance
        mask = instance_ids == instance_id
        pts = mesh_vertices[mask, :3]

        if pts.shape[0] == 0:
            continue

        # Calculate min_pts, max_pts, locations, and dimensions
        min_pts = pts.min(axis=0)
        max_pts = pts.max(axis=0)
        locations = (min_pts + max_pts) / 2
        dimensions = max_pts - min_pts

        # Store the results in instance_bboxes
        instance_bboxes[i, :3] = locations
        instance_bboxes[i, 3:6] = dimensions
        instance_bboxes[i, 6] = 1

    return instance_bboxes

def export(laz_file,
           output_file=None,
           test_mode=False):
    """Export original files to points and age_label.

    Args:
        laz_file (str): Path of the laz_file.
        output_file (str): Path of the output folder.
            Default: None.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    Returns:
        tuple: Contains the following:
            - np.ndarray: Vertices of points data.
            - int: Age label extracted from filename.
    """
    file_extension = laz_file.split('.')[-1]

    # .laz
    if file_extension == 'laz':
        with laspy.open(laz_file) as f:
            las = f.read()

        points = np.vstack((las.x, las.y, las.z)).astype(np.float64).T

    # .ply
    elif file_extension == 'ply':
        pcd = o3d.io.read_point_cloud(laz_file)
        points = np.asarray(pcd.points).astype(np.float64)
    
    else:
        raise ValueError("Unsupported file format")

    points[:, 0] -= np.mean(points[:, 0])

    points[:, 1] -= np.mean(points[:, 1])

    points[:, 2] -= np.min(points[:, 2])

    points = points.astype(np.float32)

    filename = laz_file.split('/')[-1]
    if '_' in filename:
        age_label = int(filename.split('_')[-1].replace('.laz', ''))
    else:
        age_label = -1
    #age_label = -1

    return points, age_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scan_path',
        required=True,
        help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument(
        '--label_map_file',
        required=True,
        help='path to scannetv2-labels.combined.tsv')
    parser.add_argument(
        '--scannet200',
        action='store_true',
        help='Use it for scannet200 mapping')

    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path,
                            scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(
        opt.scan_path, scan_name +
        '.txt')  # includes axisAlignment info for the train set scans.
    export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file,
           opt.output_file, scannet200=opt.scannet200)


if __name__ == '__main__':
    main()

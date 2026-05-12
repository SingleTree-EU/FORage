# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_forage_data.py
"""
import argparse
import datetime
import os
from os import path as osp

import torch
import segmentator
import open3d as o3d
import numpy as np
from load_forage_data import export
from scipy.spatial import Delaunay

DONOTCARE_CLASS_IDS = np.array([])

forage_OBJ_CLASS_IDS = np.array(
    [1])

def create_ply_with_superpoints(points, superpoints, filename):
    # Combine points and superpoints into a single array
    points_with_superpoints = np.hstack((points, superpoints[:, np.newaxis]))

    # Define the ply header
    header = f"""ply
format ascii 1.0
element vertex {points.shape[0]}
property float x
property float y
property float z
property float superpoint
end_header
"""
    # Open file and write header
    with open(filename, 'w') as f:
        f.write(header)
        # Write points and superpoints
        for point, sp in zip(points, superpoints):
            f.write(f"{point[0]} {point[1]} {point[2]} {sp}\n")
    
    print(f"Point cloud saved to {filename}")


def point_cloud_to_mesh_with_alpha_shape(points, alpha):
    # Convert numpy array to open3d point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Perform Alpha Shape
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    
    # Ensure the mesh vertices map back to the original point cloud points
    new_triangles = np.asarray(alpha_shape.triangles)
    new_vertices = points
    
    # Create a new TriangleMesh with original vertices
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Optionally compute vertex normals
    #new_mesh.compute_vertex_normals()
    
    return new_mesh

def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    forage_dir,
                    test_mode=False):
    laz_file = osp.join(forage_dir, scan_name + '.laz')
    mesh_vertices, age_label = export(
            laz_file, None, test_mode)
    
    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)

    if not test_mode:
        #assert superpoints.shape == semantic_labels.shape
        np.save(f'{output_filename_prefix}_age_label.npy', age_label)
    

def batch_export(max_num_point,
                 output_folder,
                 scan_names_file,
                 forage_dir,
                 test_mode=False
                 ):
    if test_mode and not os.path.exists(forage_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        #if osp.isfile(f'{output_filename_prefix}_vert.npy'):
        #    print('File already exists. skipping.')
        #    print('-' * 20 + 'done')
        #    continue
        try:
            export_one_scan(scan_name, output_filename_prefix, max_num_point,
                            forage_dir, test_mode)
        except Exception:
            print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./forage_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--train_forage_dir', default='train_val_data', help='forage data directory.')
    parser.add_argument(
        '--test_forage_dir',
        default='test_data',
        help='forage data directory.')
    parser.add_argument(
        '--train_scan_names_file',
        default='meta_data/train_list.txt',
        help='The path of the file that stores the train scan names.')
    parser.add_argument(
        '--val_scan_names_file',
        default='meta_data/val_list.txt',
        help='The path of the file that stores the val scan names.')
    parser.add_argument(
        '--test_scan_names_file',
        default='meta_data/test_list.txt',
        help='The path of the file that stores the test scan names.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.train_forage_dir,
        test_mode=False
        )
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.val_scan_names_file,
        args.train_forage_dir,
        test_mode=False
        )
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.test_forage_dir,
        test_mode=False
        )


if __name__ == '__main__':
    main()

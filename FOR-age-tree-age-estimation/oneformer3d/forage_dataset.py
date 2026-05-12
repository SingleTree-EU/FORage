from typing import List, Optional, Callable, Union
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmdet3d.registry import DATASETS
import os
import numpy as np


@DATASETS.register_module()
class forageDataset(BaseDataset):

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs):

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        data_infos = load(self.ann_file)['data_list']  # <- 关键修正！
        data_list = []

        for info in data_infos:
            data_list.append({
                'pts_path': os.path.join(self.data_root+'points/', info['lidar_points']['lidar_path']),
                'label_path': os.path.join(self.data_root+'age_label/', info['lidar_points']['lidar_path'])
            })

        return data_list

    def parse_data_info(self, info: dict) -> dict:
        """Convert the relative path to absolute path and set the class label."""
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])
            if 'num_pts_feats' in info['lidar_points']:
                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

        if 'age_label_path' in info:
            info['age_label_path'] = \
                osp.join(self.data_prefix.get('age_label', ''),
                         info['age_label_path'])

        return data_info

    def prepare_data(self, idx: int) -> dict:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            dict: Results passed through ``self.pipeline``.
        """
        if not self.test_mode:
            data_info = self.get_data_info(idx)
            # Pass the dataset to the pipeline during training to support mixed
            # data augmentation, such as polarmix and lasermix.
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)

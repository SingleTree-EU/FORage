import torch
import numpy as np

from mmengine.logging import MMLogger

from mmdet3d.evaluation import InstanceSegMetric
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval

from mmengine.evaluator import BaseMetric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        for eval_ann, single_pred_results in results:

            if 'originids' in single_pred_results:
                eval_ann['pts_semantic_mask'] = eval_ann['pts_semantic_mask'][single_pred_results['originids']]
                eval_ann['pts_instance_mask'] = eval_ann['pts_instance_mask'][single_pred_results['originids']]
            
            if self.metric_meta['dataset_name'] == 'S3DIS':
                pan_gt = {}
                pan_gt['pts_semantic_mask'] = eval_ann['pts_semantic_mask']
                pan_gt['pts_instance_mask'] = \
                                    eval_ann['pts_instance_mask'].copy()

                for stuff_cls in self.stuff_class_inds:
                    pan_gt['pts_instance_mask'][\
                        pan_gt['pts_semantic_mask'] == stuff_cls] = \
                                    np.max(pan_gt['pts_instance_mask']) + 1

                pan_gt['pts_instance_mask'] = np.unique(
                                                pan_gt['pts_instance_mask'],
                                                return_inverse=True)[1]
                gt_masks_pan.append(pan_gt)
            elif self.metric_meta['dataset_name'] == 'ForAINetV2':
                pan_gt = {}
                pan_gt['pts_semantic_mask'] = eval_ann['pts_semantic_mask'].copy()
                pan_gt['pts_instance_mask'] = \
                                    eval_ann['pts_instance_mask'].copy()
                thing_min_value = min(self.thing_class_inds)
                for stuff_cls in self.stuff_class_inds:
                    pan_gt['pts_instance_mask'][\
                        pan_gt['pts_semantic_mask'] == stuff_cls] = \
                                    np.max(pan_gt['pts_instance_mask']) + 1

                pan_gt['pts_instance_mask'] = np.unique(
                                                pan_gt['pts_instance_mask'],
                                                return_inverse=True)[1]
                for thing_cls in self.thing_class_inds:
                    pan_gt['pts_semantic_mask'][pan_gt['pts_semantic_mask'] == thing_cls] = thing_min_value

                gt_masks_pan.append(pan_gt)
            else:
                gt_masks_pan.append(eval_ann)
            
            pred_masks_pan.append({
                'pts_instance_mask': \
                    single_pred_results['pts_instance_mask'][1],
                'pts_semantic_mask': \
                    single_pred_results['pts_semantic_mask'][1]
            })

            gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])            
            pred_semantic_masks_sem_task.append(
                single_pred_results['pts_semantic_mask'][0])

            if self.metric_meta['dataset_name'] == 'S3DIS':
                gt_semantic_masks_inst_task.append(eval_ann['pts_semantic_mask'])
                gt_instance_masks_inst_task.append(eval_ann['pts_instance_mask'])  
            elif self.metric_meta['dataset_name'] == 'ForAINetV2':
                pan_gt = {}
                pan_gt['pts_semantic_mask'] = eval_ann['pts_semantic_mask'].copy()
                pan_gt['pts_instance_mask'] = \
                                    eval_ann['pts_instance_mask'].copy()
                thing_min_value = min(self.thing_class_inds)
                for stuff_cls in self.stuff_class_inds:
                    pan_gt['pts_instance_mask'][\
                        pan_gt['pts_semantic_mask'] == stuff_cls] = \
                                    np.max(pan_gt['pts_instance_mask']) + 1

                pan_gt['pts_instance_mask'] = np.unique(
                                                pan_gt['pts_instance_mask'],
                                                return_inverse=True)[1]
                for thing_cls in self.thing_class_inds:
                    pan_gt['pts_semantic_mask'][pan_gt['pts_semantic_mask'] == thing_cls] = thing_min_value

                gt_semantic_masks_inst_task.append(pan_gt['pts_semantic_mask'])
                gt_instance_masks_inst_task.append(pan_gt['pts_instance_mask'])  
            else:
                sem_mask, inst_mask = self.map_inst_markup(
                    eval_ann['pts_semantic_mask'].copy(), 
                    eval_ann['pts_instance_mask'].copy(), 
                    self.valid_class_ids[num_stuff_cls:],
                    num_stuff_cls)
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)           
            
            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(
                torch.tensor(single_pred_results['instance_labels']))
            pred_instance_scores.append(
                torch.tensor(single_pred_results['instance_scores']))

        if self.metric_meta['dataset_name'] == 'ForAINetV2':
            pan_class = ['nontree','tree']
            pan_stuff_class_inds=[0] 
            pan_thing_class_inds=[1]
            pan_thing_classes = [pan_class[i] for i in pan_thing_class_inds]
            pan_stuff_classes = [pan_class[i] for i in pan_stuff_class_inds]
            pan_label2cat = {i: name for i, name in enumerate(pan_class)}
            ret_pan = panoptic_seg_eval(
                gt_masks_pan, pred_masks_pan, pan_class, pan_thing_classes,
                pan_stuff_classes, self.min_num_points, self.id_offset,
                pan_label2cat, ignore_index, logger)
        else:
            ret_pan = panoptic_seg_eval(
            gt_masks_pan, pred_masks_pan, classes, thing_classes,
            stuff_classes, self.min_num_points, self.id_offset,
            label2cat, ignore_index, logger)

        #ret_sem = seg_eval(
        #    gt_semantic_masks_sem_task,
        #    pred_semantic_masks_sem_task,
        #   label2cat,
        #    ignore_index[0],
        #    logger=logger)
        max_value = max(max(self.thing_class_inds), max(self.stuff_class_inds))+1
        if not ignore_index:
            ret_sem = seg_eval(
                gt_semantic_masks_sem_task,
                pred_semantic_masks_sem_task,
                label2cat,
                max_value,
                logger=logger)
        else:
            ret_sem = seg_eval(
                gt_semantic_masks_sem_task,
                pred_semantic_masks_sem_task,
                label2cat,
                ignore_index[0],
                logger=logger)

        if self.metric_meta['dataset_name'] == 'S3DIS':
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids,
                class_labels=classes[:-1],
                logger=logger)
        elif self.metric_meta['dataset_name'] == 'ForAINetV2':
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:-1], #self.valid_class_ids[num_stuff_cls:],
                class_labels=['tree'], #classes[num_stuff_cls:],
                logger=logger)
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger)

        metrics = dict()
        for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def map_inst_markup(self,
                        pts_semantic_mask,
                        pts_instance_mask,
                        valid_class_ids,
                        num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes
        
        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask


@METRICS.register_module()
class InstanceSegMetric_(InstanceSegMetric):
    """The only difference with InstanceSegMetric is that following ScanNet
    evaluator we accept instance prediction as a boolean tensor of shape
    (n_points, n_instances) instead of integer tensor of shape (n_points, ).

    For this purpose we only replace instance_seg_eval call.
    """

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta['classes']
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            gt_instance_masks.append(eval_ann['pts_instance_mask'])
            pred_instance_masks.append(
                single_pred_results['pts_instance_mask'])
            pred_instance_labels.append(single_pred_results['instance_labels'])
            pred_instance_scores.append(single_pred_results['instance_scores'])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger)

        return ret_dict

@METRICS.register_module()
class ForSpeciesClsMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preds = []
        self.labels = []

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            gt_label = data_sample['gt_pts_seg']['species_label'].cpu().numpy().item()
            pred_label = data_sample['pred_species_label']

            self.results.append((gt_label, pred_label))

    def compute_metrics(self, results):
        gt_labels, pred_labels = zip(*results)

        accuracy = accuracy_score(gt_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_labels, pred_labels, average='weighted', zero_division=0)

        return {
            'OA': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

@METRICS.register_module()
class ForAgeRegMetric(BaseMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        gt_labels = []
        pred_labels = []

        for data_sample in data_samples:
            gt_label = data_sample['gt_pts_seg']['age_label'].cpu().numpy().item()
            pred_label = data_sample['pred_age_label']

            gt_labels.append(gt_label)
            pred_labels.append(pred_label)

        # 关键是这里，必须返回成dict形式
        self.results.append({
            'gt_labels': gt_labels,
            'pred_labels': pred_labels
        })

    def compute_metrics(self, results):
        #logger: MMLogger = MMLogger.get_current_instance()

        gt_labels_all = []
        pred_labels_all = []

        for result in results:
            gt_labels_all.extend(result['gt_labels'])
            pred_labels_all.extend(result['pred_labels'])

        gt_labels_all = np.array(gt_labels_all, dtype=np.float32)
        pred_labels_all = np.array([
            p.item() if isinstance(p, torch.Tensor) else p
            for p in pred_labels_all
        ], dtype=np.float32)

        mae = float(mean_absolute_error(gt_labels_all, pred_labels_all))
        mse = float(mean_squared_error(gt_labels_all, pred_labels_all))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(gt_labels_all, pred_labels_all))
        mbe = float(np.mean(pred_labels_all - gt_labels_all))

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MBE': mbe
        }

@METRICS.register_module()
class FordbhRegMetric(BaseMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        gt_labels = []
        pred_labels = []

        for data_sample in data_samples:
            gt_label = data_sample['gt_pts_seg']['dbh_label'].cpu().numpy().item()
            pred_label = data_sample['pred_dbh_label']

            gt_labels.append(gt_label)
            pred_labels.append(pred_label)

        # 关键是这里，必须返回成dict形式
        self.results.append({
            'gt_labels': gt_labels,
            'pred_labels': pred_labels
        })

    def compute_metrics(self, results):
        #logger: MMLogger = MMLogger.get_current_instance()

        gt_labels_all = []
        pred_labels_all = []

        for result in results:
            gt_labels_all.extend(result['gt_labels'])
            pred_labels_all.extend(result['pred_labels'])

        gt_labels_all = np.array(gt_labels_all, dtype=np.float32)
        pred_labels_all = np.array([
            p.item() if isinstance(p, torch.Tensor) else p
            for p in pred_labels_all
        ], dtype=np.float32)

        mae = float(mean_absolute_error(gt_labels_all, pred_labels_all))
        mse = float(mean_squared_error(gt_labels_all, pred_labels_all))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(gt_labels_all, pred_labels_all))
        mbe = float(np.mean(pred_labels_all - gt_labels_all))

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MBE': mbe
        }

@METRICS.register_module()
class ForheightRegMetric(BaseMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        gt_labels = []
        pred_labels = []

        for data_sample in data_samples:
            gt_label = data_sample['gt_pts_seg']['height_label'].cpu().numpy().item()
            pred_label = data_sample['pred_height_label']

            gt_labels.append(gt_label)
            pred_labels.append(pred_label)

        self.results.append({
            'gt_labels': gt_labels,
            'pred_labels': pred_labels
        })

    def compute_metrics(self, results):
        #logger: MMLogger = MMLogger.get_current_instance()

        gt_labels_all = []
        pred_labels_all = []

        for result in results:
            gt_labels_all.extend(result['gt_labels'])
            pred_labels_all.extend(result['pred_labels'])

        gt_labels_all = np.array(gt_labels_all, dtype=np.float32)
        pred_labels_all = np.array([
            p.item() if isinstance(p, torch.Tensor) else p
            for p in pred_labels_all
        ], dtype=np.float32)

        mae = float(mean_absolute_error(gt_labels_all, pred_labels_all))
        mse = float(mean_squared_error(gt_labels_all, pred_labels_all))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(gt_labels_all, pred_labels_all))
        mbe = float(np.mean(pred_labels_all - gt_labels_all))

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MBE': mbe
        }

@METRICS.register_module()
class ForMultiTaskMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #self.results_species = []
        #self.results_age = []
        #self.results_dbh = []
        #self.results_height = []

    def process(self, data_batch, data_samples):
        results_species = []
        results_age = []
        results_dbh = []
        results_height = []

        for data_sample in data_samples:
            if 'species_label' in data_sample['gt_pts_seg']:
                gt_label_species = data_sample['gt_pts_seg']['species_label'].cpu().numpy().item()
                pred_label_species = data_sample['pred_species_label']
                results_species.append((gt_label_species, pred_label_species))

            if 'age_label' in data_sample['gt_pts_seg']:
                gt_label_age = data_sample['gt_pts_seg']['age_label'].cpu().numpy().item()
                pred_label_age = data_sample['pred_age']
                results_age.append((gt_label_age, pred_label_age))

            if 'dbh_label' in data_sample['gt_pts_seg']:
                gt_label_dbh = data_sample['gt_pts_seg']['dbh_label'].cpu().numpy().item()
                pred_label_dbh = data_sample['pred_dbh']
                results_dbh.append((gt_label_dbh, pred_label_dbh))

            if 'height_label' in data_sample['gt_pts_seg']:
                gt_label_height = data_sample['gt_pts_seg']['height_label'].cpu().numpy().item()
                pred_label_height = data_sample['pred_height']
                results_height.append((gt_label_height, pred_label_height))

        self.results.append({
            'species': results_species,
            'age': results_age,
            'dbh': results_dbh,
            'height': results_height,
        })
        #.append({
        #    'species': list(self.results_species), 
        #    'age': list(self.results_age),
        #    'dbh': list(self.results_dbh),
        #    'height': list(self.results_height),
        #})


    def compute_metrics(self, results):
        metrics = {}

        all_species = []
        all_age = []
        all_dbh = []
        all_height = []

        for result in results:
            all_species.extend(result['species'])
            all_age.extend(result['age'])
            all_dbh.extend(result['dbh'])
            all_height.extend(result['height'])

        print(f"Total samples - Species: {len(all_species)}, Age: {len(all_age)}, DBH: {len(all_dbh)}, Height: {len(all_height)}")

        if all_species:
            gt_labels, pred_labels = zip(*all_species)
            accuracy = accuracy_score(gt_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                gt_labels, pred_labels, average='weighted', zero_division=0)
            print(f"Species - OA: {accuracy:.4f}, F1: {f1:.4f}")
            metrics.update({
                'Species_OA': accuracy,
                'Species_Precision': precision,
                'Species_Recall': recall,
                'Species_F1': f1
            })

        def compute_regression_metrics(results_list, task_name):
            if not results_list:
                print(f"No results for {task_name}.")
                return {}

            gt_labels, pred_labels = zip(*results_list)

            gt_labels = np.array(gt_labels, dtype=np.float32)
            pred_labels = np.array([
                p.item() if isinstance(p, torch.Tensor) else p
                for p in pred_labels
            ], dtype=np.float32)

            mae = float(mean_absolute_error(gt_labels, pred_labels))
            mse = float(mean_squared_error(gt_labels, pred_labels))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(gt_labels, pred_labels))
            mbe = float(np.mean(pred_labels - gt_labels))

            print(f"{task_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MBE: {mbe:.4f}")

            return {
                f'{task_name}_MAE': mae,
                f'{task_name}_MSE': mse,
                f'{task_name}_RMSE': rmse,
                f'{task_name}_R2': r2,
                f'{task_name}_MBE': mbe
            }

        metrics.update(compute_regression_metrics(all_age, 'Age'))
        metrics.update(compute_regression_metrics(all_dbh, 'DBH'))
        metrics.update(compute_regression_metrics(all_height, 'Height'))

        return metrics

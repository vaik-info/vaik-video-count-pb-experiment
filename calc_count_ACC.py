import argparse
import os
import glob
import json
import numpy as np


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index=0):
    total_area_intersect = np.zeros((num_classes,), dtype=np.float32)
    total_area_union = np.zeros((num_classes,), dtype=np.float32)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float32)
    total_area_label = np.zeros((num_classes,), dtype=np.float32)
    for i in range(len(results)):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(np.asarray(results[i]), np.asarray(gt_seg_maps[i]), num_classes, ignore_index=-1)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union

    mask = np.ones(iou.shape, dtype=bool)
    mask[ignore_index] = False
    return np.mean(iou[mask]), acc, iou


def calc_count_ACC(json_dict_list, classes):
    acc_dict = {}
    exclude_zero_acc_dict = {}
    for key in classes:
        acc_dict[key] = []
        exclude_zero_acc_dict[key] = []
    for json_dict in json_dict_list:
        for target_class_label in classes:
            answer_count = json_dict['answer'][target_class_label] if target_class_label in json_dict['answer'].keys() else 0
            pred_count = round(json_dict['count'][classes.index(target_class_label)])
            if answer_count > 0:
                count_error = abs(answer_count - pred_count) / answer_count
            elif pred_count > 0:
                count_error = abs(answer_count - pred_count) / pred_count
            else:
                count_error = 0.
            count_acc = 1. - count_error
            acc_dict[target_class_label].append(count_acc)

            if not(answer_count == 0 and pred_count == 0):
                exclude_zero_acc_dict[target_class_label].append(count_acc)
    mean_acc_list = [np.mean(acc_dict[label]) for label in classes]
    mean_exclude_zero_acc_list = [np.mean(exclude_zero_acc_dict[label]) for label in classes]
    print(f'CountACCRatio[all]:{np.mean(mean_acc_list):.4f}')
    for index, label in enumerate(classes):
        print(f'{label}: {mean_acc_list[index]:.4f}')
    print()

    print(f'CountACCRatio[exclude_zero_both]:{np.mean(mean_exclude_zero_acc_list):.4f}')
    for index, label in enumerate(classes):
        print(f'{label}: {mean_exclude_zero_acc_list[index]:.4f}')

def main(input_json_dir_path, input_classes_path):
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    json_path_list = glob.glob(os.path.join(input_json_dir_path, '*.json'))
    json_dict_list = []
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            json_dict_list.append(json_dict)
    calc_count_ACC(json_dict_list, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_json_dir_path', type=str, default=os.path.expanduser('~/.vaik-video-count-pb-experiment/test_dataset_out'))
    parser.add_argument('--input_classes_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_dataset/classes.txt'))
    args = parser.parse_args()

    args.input_json_dir_path = os.path.expanduser(args.input_json_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)

    main(**args.__dict__)
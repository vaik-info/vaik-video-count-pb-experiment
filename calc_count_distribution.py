import argparse
import os
import glob
import json
import numpy as np
from sklearn.metrics import mean_squared_error


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
    distribution = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union

    mask = np.ones(iou.shape, dtype=bool)
    mask[ignore_index] = False
    return np.mean(iou[mask]), distribution, iou


def calc_count_distribution_mse(json_dict_list, classes, skip_frame):
    distribution_dict = {}
    for key in classes:
        distribution_dict[key] = []
    for json_dict in json_dict_list:
        answer_count_list = json_dict['answer']['detail']
        pred_count = np.asarray(json_dict['cam']['array']).reshape(json_dict['cam']['shape'])
        print(f'video_path: {os.path.basename(json_dict["video_path"])}')
        mse_log_string = ""
        answer_count_list_string = "Answer:\t"
        pred_count_list_string = "Pred:\t"
        for target_class_label in classes:
            target_count_list = [0, ] * len(answer_count_list[0]['count'])
            for answer_count in answer_count_list:
                if answer_count['label'] == target_class_label:
                    count_list = [1 if count else 0 for count in answer_count['count']]
                    total_count_list = sum(count_list)
                    count_list = [count / total_count_list for count in count_list]
                    for count_index, count in enumerate(count_list):
                        target_count_list[count_index] += count
            target_count_list = target_count_list[::skip_frame]
            target_pred_count = pred_count[::skip_frame, :, :, classes.index(target_class_label)]
            pred_count_list = np.sum(target_pred_count, axis=-1)
            pred_count_list = np.sum(pred_count_list, axis=-1).tolist()
            mse = mean_squared_error(target_count_list, pred_count_list)
            distribution_dict[target_class_label].append(mse)
            mse_log_string += f'{target_class_label}: {json_dict["answer"][target_class_label] if target_class_label in json_dict["answer"].keys() else 0 }[count(answer)]-{sum(pred_count_list):.2f}[count(pred)]-{mse:.4f}[mse], '
            answer_count_list_string += f'{target_class_label}: {[f"{elem:.2f}" for elem in target_count_list]}, '
            pred_count_list_string += f'{target_class_label}: {[f"{elem:.2f}" for elem in pred_count_list]}, '

        print(mse_log_string)
        print(answer_count_list_string)
        print(pred_count_list_string)
        print()
    mean_acc_list = [np.mean(distribution_dict[label]) for label in classes]
    print(f'CountDistributionMSE[all]:{np.mean(mean_acc_list):.4f}')
    for index, label in enumerate(classes):
        print(f'{label}: {mean_acc_list[index]:.4f}')
    print()

def main(input_json_dir_path, input_classes_path, skip_frame):
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    json_path_list = sorted(glob.glob(os.path.join(input_json_dir_path, '*.json')))
    json_dict_list = []
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            json_dict_list.append(json_dict)
    calc_count_distribution_mse(json_dict_list, classes, skip_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_json_dir_path', type=str, default=os.path.expanduser('~/.vaik-video-count-pb-experiment/test_dataset_out'))
    parser.add_argument('--input_classes_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_dataset/classes.txt'))
    parser.add_argument('--skip_frame', type=int, default=1)
    args = parser.parse_args()

    args.input_json_dir_path = os.path.expanduser(args.input_json_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)

    main(**args.__dict__)
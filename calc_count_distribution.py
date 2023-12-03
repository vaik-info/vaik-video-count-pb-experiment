import argparse
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def dump(json_dict, classes, answer_count_list, pred_count, skip_frame, prefix):
    print(f'[{prefix}]video_path: {os.path.basename(json_dict["video_path"])}')
    mse_log_string = ""
    answer_count_list_string = "Answer:\t"
    pred_count_list_string = "Pred:\t"
    mse_list = []
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
        mse_list.append(mse)
        mse_log_string += f'{target_class_label}: {json_dict["answer"][target_class_label] if target_class_label in json_dict["answer"].keys() else 0 }[count(answer)]-{sum(pred_count_list):.4f}[count(pred)]-{mse:.4f}[mse], '
        answer_count_list_string += f'{target_class_label}: {[f"{elem:.4f}" for elem in target_count_list]}, '
        pred_count_list_string += f'{target_class_label}: {[f"{elem:.4f}" for elem in pred_count_list]}, '
    print(mse_log_string)
    print(answer_count_list_string)
    print(pred_count_list_string)
    print()
    return mse_list

def calc_count_distribution_mse(json_dict_list, classes, skip_frame):
    cam_mse_list = []
    grad_mse_list_list = []
    for _ in json_dict_list[0]['grad_cam']:
        grad_mse_list_list.append([])
    for json_dict in json_dict_list:
        answer_count_list = json_dict['answer']['detail']
        pred_count = np.asarray(json_dict['cam']['array']).reshape(json_dict['cam']['shape'])
        cam_mse = dump(json_dict, classes, answer_count_list, pred_count, skip_frame, 'cam')
        cam_mse_list.append(cam_mse)
        for grad_index, grad_cam in enumerate(json_dict['grad_cam']):
            pred_count = np.asarray(grad_cam['array']).reshape(grad_cam['shape'])
            grad_mse_list = dump(json_dict, classes, answer_count_list, pred_count, skip_frame, f'grad{grad_index:02d}')
            grad_mse_list_list[grad_index].append(grad_mse_list)
    for pred_index in range(len(cam_mse_list[0])):
        cam_mse_pred = np.mean([cam_mse[pred_index] for cam_mse in cam_mse_list])
        print(f'{classes[pred_index]}, {cam_mse_pred:.4f}[cam], ', end='')
        for grad_index, grad_mse_list in enumerate(grad_mse_list_list):
            grad_mse_pred = np.mean([grad_mse_pred[pred_index] for grad_mse_pred in grad_mse_list])
            print(f'{grad_mse_pred:.4f}[grad{grad_index:02d}], ', end='')
        print()
    print(f'MSE, {np.mean(cam_mse_list):.4f}[cam], ', end='')
    for grad_index, grad_mse_list in enumerate(grad_mse_list_list):
        print(f'{np.mean(grad_mse_list):.4f}[grad{grad_index:02d}], ', end='')
def main(input_json_dir_path, input_classes_path, skip_frame):
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    json_path_list = sorted(glob.glob(os.path.join(input_json_dir_path, '*.json')))
    json_dict_list = []
    for json_path in tqdm(json_path_list):
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
import argparse
import os
import glob
import json
import imageio
import time
import numpy as np
from vaik_video_count_pb_inference.pb_model import PbModel


def main(input_saved_model_dir_path, input_classes_path, input_data_dir_path, output_dir_path, skip_frame):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    model = PbModel(input_saved_model_dir_path, classes)

    types = ('*.avi', '*.mp4')
    video_path_list = []
    for file in types:
        video_path_list.extend(glob.glob(os.path.join(input_data_dir_path, f'{file}'), recursive=True))

    total_inference_time = 0
    total_frames_num = 0

    for video_path in video_path_list:
        video = imageio.get_reader(video_path,  'ffmpeg')
        frames = [frame for frame in video][::skip_frame]
        frames = np.stack(frames, axis=0)
        total_frames_num += len(frames)

        start = time.time()
        outputs, raw_pred = model.inference([frames])
        end = time.time()
        total_inference_time += (end - start)
        output = outputs[0]
        with open(f'{os.path.splitext(video_path)[0]}.json', 'r') as f:
            json_dict = json.load(f)
        output_json_path = os.path.join(output_dir_path, os.path.splitext(os.path.basename(video_path))[0] + '.json')
        output['label'] = classes
        output['answer'] = json_dict
        output['video_path'] = video_path
        output['skip_frame'] = skip_frame
        output['cam'] = {'array': [round(elem, 5) for elem in output['cam'].flatten().tolist()], 'shape': output['cam'].shape}
        with open(output_json_path, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    print(f'{len(video_path_list)/total_inference_time}[videos/sec]')
    print(f'{total_frames_num/total_inference_time}[frame/sec]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_dir_path', type=str,
                        default='/home/kentaro/.vaik-video-count-pb-trainer/output_model/2023-12-02-03-51-31/step-1000_batch-4_epoch-19_loss_0.0912_active_huber_loss_0.0912_blank_huber_loss_0.0001_val_loss_0.1162_val_active_huber_loss_0.1161_val_blank_huber_loss_0.0001')
    parser.add_argument('--input_classes_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_dataset/classes.txt'))
    parser.add_argument('--input_data_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_dataset/data'))
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-video-count-pb-experiment/test_dataset_out')
    parser.add_argument('--skip_frame', type=int, default=1)
    args = parser.parse_args()

    args.input_saved_model_dir_path = os.path.expanduser(args.input_saved_model_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_data_dir_path = os.path.expanduser(args.input_data_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
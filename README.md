# vaik-video-count-pb-experiment

Create json file by count for video model. Calc count ACC.

![video-count](https://github.com/vaik-info/vaik-video-count-pb-experiment/assets/116471878/59263d64-5d32-48ac-b387-31f025333597)

## Install

```shell
pip install -r requirements.txt
```

## Usage

-------

### Create json file

```shell
python inference.py --input_saved_model_dir_path '~/.vaik-video-count-pb-trainer/output_model/2023-11-28-07-10-44/step-1000_batch-8_epoch-34_loss_0.1202_val_loss_0.1023' \
                --input_classes_path './test_dataset/classes.txt' \
                --input_data_dir_path './test_dataset/data' \
                --output_dir_path '~/.vaik-video-count-pb-experiment/test_dataset_out' \
                --skip_frame 2
```

- input_data_dir_path

```shell
├── valid_000000000_raw.avi
├── valid_000000000_raw.json
├── valid_000000001_raw.avi
├── valid_000000001_raw.json
├── valid_000000002_raw.avi
・・・
```

#### Output
- output_dir_path
    - example

```shell
 {
    "answer": {
        "one": 1,
        "zero": 1
    },
    "cam": {
        "array": [[
            0.0,
	・・・
        ]],
        "shape": [
            241,
            265,
            3
        ]
    },
    "count": [
        1,
        1,
        0
    ],
    "video_path": "/home/kentaro/GitHub/vaik-count-pb-experiment/test_dataset/data/valid_000000000_raw.avi",
    "label": [
        "zero",
        "one",
        "two"
    ]
}
```

--------

### Calc count ACC

```shell
python calc_count_ACC.py --input_json_dir_path '~/.vaik-video-count-pb-experiment/test_dataset_out' \
                --input_classes_path './test_dataset/classes.txt'
```

#### Output

```shell
CountACCRatio[all]:0.9500
zero: 0.9500
one: 0.9000
two: 1.0000

CountACCRatio[exclude_zero_both]:0.9444
zero: 0.9444
one: 0.8889
two: 1.0000
```

-----------

### Draw

```shell
python draw.py --input_json_dir_path '~/.vaik-video-count-pb-experiment/test_dataset_out' \
                --input_classes_path './test_dataset/classes.txt' \
                --output_dir_path '~/.vaik-video-count-pb-experiment/test_dataset_out_draw'
```

#### Output

![video-count](https://github.com/vaik-info/vaik-video-count-pb-experiment/assets/116471878/59263d64-5d32-48ac-b387-31f025333597)
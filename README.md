# RAPIER

Paper: Low-Quality Training Data Only? A Robust Framework for Detecting Encrypted Malicious Network Traffic

## Preprocessing:

```
cd ./Preprocess
python Feature_Extract.py "input_dir" "sequence_data_path" "ext"
python get_origin_flow_data.py "sequence_data_path" "save_dir" "data_type"
```

* `input_dir: ` The directory of all raw pcap files of one kind of data, e.g. `benign`, `malicious`, or `test`.
* `sequence_data_path: ` The sequence data of all flows in pcap files, without zero-padding. (The values are prefix-cumulative values, and will be further processed by `get_origin_flow_data.py`)
* `ext: ` The extension name of pcap files to process (e.g. `pcap`, `pcapng`).
* `save_dir: ` The save directory of the processed sequence data of all flows.
* `data_type: ` The kind of the processed data, e.g. `w`, `b` and `test`.

Output: A sequence numpy file of `data_type` in `save_dir`. i.e., `{save_dir}/{data_type}.npy`, the dimension of each sample is 50. Need to add the 51st dimension for detection.

## Detection/Prediction:

```
cd ./main
python main.py
```

The argument can be modified in `main.py` are:
* `data_dir: ` The directory of all sequence data.
* `feat_dir: ` The directory of all feature data.
* `made_dir: ` The directory of all results calculated by `MADE`.
* `model_dir: ` The directory of all trained models.
* `result_dir: ` The directory of the detection/prediction result of the test data.

The required input files:
* `{data_dir}/{benign.npy}: ` The benign preprocessed training data. 
* `{data_dir}/{malicious.npy}: ` The malicious preprocessed training data.
* `{data_dir}/{test.npy}: ` The preprocessed testing data.

All data in `{data_dir}` should have dimensions of (*n*, 51), where *n* is the number of samples. Each sample is a 51-dimension vector, where first 50 dimensions are time-series data of traffic and the last one is the true label for the sample (used to evaluate. `0` is for benign and `1` is for malicious). If RAPIER is used to predict, the last dimension can be any value.

Output:
* `{result_dir}/{prediction.npy}: ` The prediction of all testing data. `1` is malicious and `0` is benign.

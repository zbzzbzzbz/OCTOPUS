## Environmental Settings
Our experiments are conducted on Ubuntu 22.04, two NVIDIA 4090 GPUs, 128GB RAM, and Intel i9-14700K. OCTOPUS is implemented by `Python 3.10`, `Cuda 12.1`.

```shell
# create virtual environment
conda create --name OCTOPUS python=3.10 cudatoolkit=12.1

# activate virtual environment
conda activate OCTOPUS

# install other dependencies
pip install -r requirements.txt
```

## Usage

Here we take the sample of WebVid-COVR dataset as an example to demonstrate the usage.


### WebVid_CoVR demo
We provide a demo retrieved on the WebVid-COVR-Test dataset (Assuming you have already set up the environment)

Step 1: download clip model weights
```shell
cd OCTOPUS/model/openai
python model_download.py
```

Step 2: using src/main.py to execute Bi-Retrieval
```shell
cd OCTOPUS/
python src/main.py
```


### File Directory Structure

```
OCTOPUS/
├── datasets/
│   ├── WebVid_CoVR/
│   ├── FineCVR/
│   ├── FASHIONIQ/
│   ├── circo/
│   └── CIRR/
├── model/
├── config/
└── src/
└── tools/
```
### WebVid_CoVR
Download the WebVid-CoVR—Test dataset following the instructions in the [**official repository**](CoVR: Learning Composed Video Retrieval from Web Video Captions).
After downloading the dataset, ensure that the folder structure matches the following:

```
WebVid_CoVR/
├── annotations/
│   ├── webvid8m-covr_test.csv
└── 8M/
    └── train/
        ├── 0
        ├── 1
        └── ...
```

### Preprocess

Step 1: generate target description:
Use tools/generate_tar_desc_webvid.py to generate target description
```shell
python generate_tar_desc_webvid.py
```

Step 2: generate visual proxy:
Use tools/generate_tar_image.py to generate visual proxy
```shell
bash generate_tar_image_webvid
```

Step 3: generate textual proxies:
Use tools/object_detection_webvid.py to detect the object
Use tools/enrich_tar_desc_webvid.py to generate textual proxies:
```shell
python object_detection_webvid.py
python enrich_tar_desc_webvid.py
```

After finishing the Preprocess, ensure that the folder structure matches the following:

```
WebVid_CoVR/
├── annotations/
│   ├── webvid8m-covr_test.csv
└── 8M/
    └── train/
        ├── 0
        ├── 1
        └── ...
├── preload/
│   ├── gen_tar/
│   ├── gen_tar_desc/
│   ├── obj_det/
```

### Retrieval
Using src/main.py to execute Bi-Retrieval and Re-rank
```shell
python src/main.py
```
##  Image Orientation Detection Using ResNet - Batch Processing

Forked from this repo: https://github.com/parsapoorsh/resnet-ixion

## Intro

this model is a fine-tuned variant of resnet152.

it's been converted to a classfication model with 4 classes. `0°, 90°, 180°, 270°`.

## Batch Folder Processing

1. The `batch_orientation_detector.py` tool loads the pre-trained ResNet152 model
2. For each image, it:
  - Creates preprocessed tensor versions
  - each of the four possible orientations
  - Adjusts predictions based on the applied rotation
  - Combines and averages probabilities
  - Determines the most likely correct orientation
3. Optionally corrects and saves rotated images

### Features

- Recursive folder scanning - Processes all images in a folder and its subfolders
- Multi-angle analysis - Creates and analyzes rotated versions (0°, 90°, 180°, 270°) of each image
- Orientation averaging - Combines predictions from all rotations for more accurate results
- Automatic correction - Optionally rotates images to their correct orientation
- CSV reporting - Saves detailed results to a CSV file
- GPU acceleration - Uses CUDA if available for faster processing

### Usage

1. Install Nvidia GameReady GPU Driver: https://www.nvidia.com/de-de/geforce/game-ready-drivers/
2. Install Nvidia CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Install Nvidia cuDNN Libraries: https://developer.nvidia.com/cudnn-downloads

`python batch_orientation_detector.py input_folder [options]`

### Example

```bash
# Clone the repo & setup requirements (using Python 3.11)
git clone https://github.com/hexxone/resnet-ixion.git
pip install -r .\requirement.txt

# Download the model if needed
wget https://github.com/parsapoorsh/resnet-ixion/releases/download/1.0.0/resnet152_ixion_e3-84529282.onnx -O resnet152_ixion_e3-84529282.onnx

# Process images with automatic correction
python batch_orientation_detector.py ~/Pictures/vacation --correct --output_folder ~/Pictures/corrected

# Process and save detailed results
python batch_orientation_detector.py ~/Pictures/scans --verbose --csv results.csv
```

## Licence

same licence as resnet. `apache-2.0`

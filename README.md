# Mediapipe Two-Stage Gesture Training

This workspace contains a two-stage gesture recognition pipeline built around a detector (RGB frames) and a classifier fed by either MediaPipe landmarks or YOLO + MobileNet embeddings. YOLO + MobileNet is still being trained, the models, real-time test and quantization will be posted later.

## 1. Environment setup

```powershell
# Clone the repository first, create your virtual enviorment, then run:
pip install -r requirements.txt
```

> The project expects Python 3.12 and PyTorch with CUDA support. Adjust the installation command for your GPU/driver stack if necessary.

## 2. Dataset layout

Link to the dataset and information: https://gibranbenitez.github.io/IPN_Hand/

```
annotations/
  Annot_TrainList.txt
  Annot_TestList.txt
  classIdx.txt
  metadata.csv
  ...
data/videos/
  *.avi
cache/
  mediapipe/
  yolo_mobilenet/
```

Each annotation entry references a video (without the `.avi` suffix) plus the gesture class and start/end frames. The cache folders are populated automatically with pre-computed features for their respective trainings. I highly recommend precomputing features as it drastically speeds up the model training (100x) at the cost of some additional setup time.

## 3. Training the models

Key switches:

- `--skip-detector` / `--skip-classifier` control which stage runs.
- `--feature-type` chooses between `mediapipe` (landmarks) and `yolo_mobilenet` embeddings.
- `--max-train-samples` and `--max-val-samples` cap the number of windows for smoke tests. 
- `--features-cache-dir` to use the recommended pre-computed features
- `--max-classifier-epochs` and `--max-detector-epochs` to set the number of epochs

An example CLI command I used for training was: 
```powershell
python train_two_stage.py --temporal-model gru --num-layers 3 --hidden-size 256 --max-classifier-epochs 50 --skip-detector --feature-type mediapipe --features-cache-dir cache/mediapipe --optimizer adam
```

The script uses PyTorch Lightning. Checkpoints (and TensorBoard logs) are written under `models/` and `lightning_logs/` respectively.

## 4. To be Added

This contains a list of things that will be added in the near future:

- Feature precompute scripts
- Data Visualization and Analysis
- Link to download all of the trained models
- Real-time test on Desktop/Laptop
- Quantization/Export Scripts for SNPE deployment

## 5. Acknowledgements

This project builds on the IPN Hand dataset. Please cite the original work if you use this repository in your research:

```
@inproceedings{bega2020IPNhand,
  title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
  author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
  booktitle={25th International Conference on Pattern Recognition, {ICPR 2020}, Milan, Italy, Jan 10--15, 2021},
  pages={4340--4347},
  year={2021},
  organization={IEEE}
}
```


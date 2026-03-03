# BlazeFace-PyTorch (STM32MP257F-DK)

## A quoi sert le projet

Ce projet fournit une implementation **PyTorch** de BlazeFace, un detecteur de visage rapide et leger.
Il permet de:

- charger un modele pre-entraine (`blazeface.pth` ou `blazefaceback.pth`)
- detecter des visages et 6 points cles faciaux
- exporter le modele en **ONNX** (`export_onnx.py`)
- generer un jeu de calibration **NPZ** pour des workflows d'optimisation (`make_calib_npz.py`)

Le modele frontal est optimise pour des visages relativement proches (cas selfie), pas pour la detection de tres petits visages.

## Requirements

Environnement recommande:

- Python 3.8+
- `torch`
- `numpy`
- `opencv-python` (necessaire pour `make_calib_npz.py`)
- Jupyter (optionnel, pour les notebooks)

Installation rapide:

```bash
pip install torch numpy opencv-python notebook
```

## Comment prendre en main le projet rapidement

1. Place-toi dans le dossier du projet:

```bash
cd Face_detection/STM32MP257F-DK/BlazeFace-PyTorch
```

2. Verifie que les fichiers modeles sont presents:
- `blazeface.py`
- `blazeface.pth`
- `anchors.npy`

3. Lance le notebook de demo:

```bash
jupyter notebook Inference.ipynb
```

4. Export ONNX (optionnel):

```bash
python export_onnx.py
```

5. Generer un fichier de calibration NPZ (optionnel):
- place tes images dans `calib_images/`
- puis execute:

```bash
python make_calib_npz.py
```

## Structure projet

- `blazeface.py`: implementation du modele BlazeFace et post-traitement (NMS, decode)
- `blazeface.pth`: poids du modele frontal
- `blazefaceback.pth`: poids du modele back-camera
- `anchors.npy`: ancres pour le modele frontal
- `anchorsback.npy`: ancres pour le modele back-camera
- `export_onnx.py`: export du modele PyTorch vers ONNX
- `make_calib_npz.py`: preparation d'un dataset de calibration au format NPZ
- `Inference.ipynb`: demonstration d'inference
- `Convert.ipynb`: conversion des poids depuis la version TFLite
- `Anchors.ipynb`: generation des ancres
- `calib_images/`: dossier source d'images pour la calibration

## Ressources

- Article BlazeFace (Google Research): https://sites.google.com/view/perception-cv4arvr/blazeface
- Paper arXiv: https://arxiv.org/abs/1907.05047
- MediaPipe Face Detection: https://github.com/google/mediapipe/blob/master/mediapipe/docs/face_detection_mobile_gpu.md

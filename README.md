# Brain MRI Tumour Segmentation

Course project, Tiangong University (2022).

Semantic segmentation of tumour regions in brain MRI using a U-Net trained in TensorFlow/Keras.

## Dataset

TCGA-LGG brain MRI collection (Kaggle "LGG MRI Segmentation", `kaggle_3m`):
3,929 FLAIR slices at 256×256 with manual abnormality masks, from 110 patients.
Slices are labelled tumour / no tumour from the mask.

## Method

- Data pipeline built from raw image and mask files into a dataframe
- Random split: 2,838 training / 501 validation / 590 test slices
- On-the-fly augmentation on the training set: horizontal and vertical flips,
  small rotation, shift, shear and zoom
- U-Net (encoder 64→1024 filters, transposed-convolution decoder, batch normalisation)
- Loss: binary cross-entropy + Dice loss
- Adam, learning rate 1e-4, batch size 16, up to 30 epochs
- Callbacks: model checkpoint on best validation loss, ReduceLROnPlateau, early stopping
- Metrics: IoU and Dice coefficient

## Results (held-out test set, 590 slices)

| Metric | Score |
|---|---|
| IoU | 0.773 |
| Dice | 0.869 |

Predicted masks are compared visually against ground truth for positive and negative cases
at the end of the notebook.

## Known limitations

- The train/validation/test split is random at slice level, so slices from the same patient
  can appear in both training and test sets. A patient-level split (grouping by the
  `TCGA_*` folder) would give a more honest estimate and is the right fix.
- `BinaryCrossentropy(from_logits=True)` is applied to a sigmoid output; it should be
  `from_logits=False`. Keras warns about this during evaluation. Results were obtained
  with the setting as written.
- Single training run; no cross-validation or seed averaging.

## Files

- `Brain MRI Segmentation.ipynb` — full pipeline: data loading, visualisation,
  training, evaluation and prediction visualisation (written for Google Colab).

## How to run

1. Download the dataset from Kaggle (LGG MRI Segmentation) and unzip it so that the
   images are under `MRI/kaggle_3m/`.
2. Update the drive mount and zip path in the first cells if not using Colab.
3. Run the notebook top to bottom.

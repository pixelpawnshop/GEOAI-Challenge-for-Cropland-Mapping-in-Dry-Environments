# Cropland Mapping in Dry Environments - Zindi Challenge

This repository contains a modular pipeline for cropland mapping using Sentinel-1 and Sentinel-2 satellite data, designed for the [GeoAI Challenge for Cropland Mapping in Dry Environments](https://zindi.africa/competitions/geoai-challenge-for-cropland-mapping-in-dry-environments/data) on Zindi.

## Project Structure

```
├── data/                  # Place all competition data here (not included in repo)
│   ├── Fergana_training_samples.shp
│   ├── Orenburg_training_samples.shp
│   ├── Sentinel1.csv
│   ├── Sentinel2.csv
│   ├── Test.csv
│   ├── SampleSubmission.csv
│   └── ...
├── src/
│   ├── preprocessing/
│   │   └── join.py        # Preprocessing: feature engineering and label assignment
│   ├── modelling/
│   │   └── modelling.py   # Model training (XGBoost)
│   ├── validate/
│   │   └── validate.py    # Validation and submission file creation
│   └── pipe/
│       └── pipeline.py    # Runs the full pipeline
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── ...
```

## Setup Instructions

1. **Clone the repository**

2. **Download the competition data**
   - Register and download all required files from the [Zindi competition data page](https://zindi.africa/competitions/geoai-challenge-for-cropland-mapping-in-dry-environments/data).
   - Place all data files in the `data/` directory at the root of the project.

3. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Pipeline

From the project root, run:
```sh
python src/pipe/pipeline.py
```
This will sequentially:
- Preprocess the data and generate features (`train_features.csv`, `test_features.csv`)
- Train an XGBoost model and print validation accuracy
- Generate a submission file (`submission.csv`)

## Features Used
- **Sentinel-1**: VH, VV (aggregated: mean, std, min, max)
- **Sentinel-2**: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (aggregated: mean, std, min, max)
- Labels are assigned to each point using a KDTree nearest neighbor search from the shapefile points.

## Environment
- Python 3.11+
- See `requirements.txt` for all dependencies

## Hardware Needed
- At least 8GB RAM recommended (for large CSVs)
- CPU is sufficient for the current pipeline (XGBoost)

## Notes
- The `data/` directory is gitignored. You must download the data yourself from Zindi.
- The pipeline is modular and can be extended for deep learning/time-series models.

## Contact
For questions, open an issue or contact via Zindi discussion board.

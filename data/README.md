# Data   

Download the dataset from Kaggle:
```bash
kaggle datasets download paultimothymooney/chest-xray-pneumonia \
    --unzip -p data/chest_xray
```

Then run the re-split:
```bash
python run_pipeline.py --raw_dir data/chest_xray --run data
```

This creates `data/chest_xray_split/` with reproducible 80/10/10 train/val/test splits. 
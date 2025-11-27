### MaxViT-based Grid Prediction Model for Re-analysis Fields Using CMAQ Inputs

This repository provides an implementation of a MaxViT-based grid-to-grid prediction model designed to estimate PM2.5 grid fields using CMAQ simulation outputs as input.
The model extends the previously used class-based prediction framework into a full spatial grid prediction approach using a modified MaxViT architecture.

## Overview

The goal is to generate **re-analysis PM2.5 grid fields** from **CMAQ model outputs** by leveraging Vision Transformers.


## Model Architecture

1. **CMAQ Inputs** – chemical species and meteorological fields.  
2. **MaxViT Encoder–Decoder** – captures multi-scale spatial dependencies.  
3. **Grid Predictions** – output aligned with re-analysis grids.  
4. **Focal-R Loss** – mitigates imbalance in rare/high-impact regions.

## Dataset Requirements

External datasets required:
- **CMAQ simulation outputs**
- **Re-analysis data** (ERA5, MERRA-2, etc.)

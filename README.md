## SORA vs Reality: AI Video Comparison Pipeline

A comprehensive toolkit for analyzing and quantifying bias in AI-generated videos compared to real-world footage. This project provides a structured approach to evaluate how well SORA-generated videos reproduce reality across multiple dimensions.

## Overview

This pipeline compares original videos with their SORA-generated counterparts using state-of-the-art computer vision techniques. The goal is to quantify perceptual and semantic differences, identify potential biases, and provide detailed analysis through metrics and visualizations.

## Key Features

- Multi-dimensional video comparison using multiple metrics:
    - Structural Similarity (SSIM)
    - Perceptual similarity using LPIPS
    - Deep feature distance
    - Motion vector analysis
    - Color distribution comparison
- Semantic content analysis:
    - Object detection and classification
    - Scene transition detection
    - Depth map estimation
- Bias detection and quantification across multiple categories:
    - Gender representation
    - Ethnicity representation
    - Age groups
    - Settings (urban/rural, indoor/outdoor)
    - Activities
- Comprehensive visualizations:
    - Metric distributions
    - Correlation matrices
    - Frame-by-frame comparisons
    - Category bias plots
- Flexible configuration options
- GPU acceleration support

## Project Structure

```
SORAvs/
├── config/
│   └── config.yaml
├── data/
│   ├── original_videos/
│   ├── sora_videos/
│   └── output/
├── models/
├── src/
│   ├── feature_extraction.py
│   ├── bias_analysis.py
│   ├── video_comparison.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md

```

## Installation

```bash
# Clone the repository
git clone https://github.com/aiskakov28/SORAvs.git
cd SORAvs

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Usage

### Basic Usage

```bash
python main.py

```

This will compare videos in the default directories using the configuration in config.yaml.

### Command Line Options

```bash
python main.py --original-dir path/to/originals --sora-dir path/to/sora --output-dir path/to/output

```

### Additional Options

- `--disable-object-detection`: Skip object detection (improves stability if issues occur)
- `--max-frames N`: Set maximum number of frames to process per video
- `--use-lpips`: Enable LPIPS perceptual similarity metric
- `--gpu`: Use GPU for processing if available
- `--config path/to/config.yaml`: Use a custom configuration file

## Data Preparation

Place your original videos in the `data/original_videos/` directory and the corresponding SORA-generated videos in the `data/sora_videos/` directory. By default, the program expects matching filenames, but you can configure custom video pairs in config.yaml or via the `--pair` option.

## Configuration

The config.yaml file contains various settings for the analysis pipeline:

```yaml
# Core directories
original_dir: "data/original_videos"
sora_dir: "data/sora_videos"
output_dir: "data/output"

# Processing options
max_frames: 100
frame_sample_rate: 5
use_lpips: true
use_gpu: false
feature_model: "VGG16"

# Visualization options
plot_dpi: 300
color_palette: "coolwarm"

# Bias categories to analyze
bias_categories:
  gender: ["male", "female", "nonbinary"]
  ethnicity: ["asian", "black", "hispanic", "white"]
  age: ["child", "young", "adult", "elderly"]
  setting: ["urban", "rural", "indoor", "outdoor"]
  action: ["walking", "running", "sitting", "standing"]

```

## Output

The pipeline generates several types of output in the `data/output/` directory:

- JSON results with detailed metrics
- CSV files with summary metrics
- Visualizations of comparison results
- Frame-by-frame comparison images
- Category bias analysis plots

## Interpreting Results

### Key Metrics

- **SSIM (Structural Similarity):** 0-1 scale where 1 indicates identical structure
- **Feature Distance:** Difference in deep features (lower is better)
- **LPIPS:** Perceptual similarity (lower is better)
- **Object Recognition Accuracy:** 0-1 scale measuring semantic content match
- **Perceived Quality:** Combined metric weighting all factors

### Bias Analysis

The bias analysis shows how SORA's generation quality varies across different demographic and contextual categories. A significant difference in scores between category values may indicate bias in the generation process.

## Extending the Pipeline

You can extend the pipeline in several ways:

- Add new feature extraction methods in `feature_extraction.py`
- Implement additional bias categories in `config.yaml`
- Create custom visualizations in `visualization.py`
- Add new metrics to the `bias_analysis.py` module

## Limitations

- Object detection requires appropriate model files and may fail in complex scenes
- Depth estimation is approximated using gradient-based methods
- Bias analysis requires metadata in filenames or explicit configuration
- High computational requirements for processing many videos

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

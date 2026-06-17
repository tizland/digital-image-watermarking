# Digital Image Watermarking Using Optimal Channel Selection

## Project Description

This project presents an implementation of the paper:

**Improving Digital Image Watermarking by Means of Optimal Channel Selection**

The proposed method uses Discrete Wavelet Transform (DWT), Optimal Channel Selection, and Otsu Thresholding to embed and extract digital watermarks from color images.

## Features

- 4-Level Haar DWT
- Optimal RGB Channel Selection
- Watermark Embedding and Extraction
- Otsu Thresholding
- CPSNR Evaluation
- BER Evaluation
- Automatic Result Generation

## Dataset

The following standard images are used:

- Lena
- Peppers
- House
- Airplane

Watermark size:

- 32 × 32 pixels

Host image size:

- 512 × 512 pixels

## Project Structure

watermarking-project/

├── main.py

├── requirements.txt

├── data/

│   ├── lena.png

│   ├── peppers.png

│   ├── house.png

│   ├── airplane.png

│   └── watermark.png

├── output/

└── README.md

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Evaluation Metrics

### CPSNR

Measures the visual quality of the watermarked image.

Higher values indicate better image quality.

### BER

Measures the bit error rate during watermark extraction.

Lower values indicate more accurate watermark recovery.

## Experimental Results

| Image | CPSNR (dB) | BER | Otsu Threshold |
|--------|------------|------------|------------|
| Lena | 40.07 | 0.0039 | 36.1014 |
| Peppers | 41.79 | 0.5273 | 70.3383 |
| House | 40.59 | 0.5830 | 63.7074 |
| Airplane | 38.74 | 0.7783 | 44.7218 |

## Author

Mohammad Mahdi Bagheri

M.Sc. Student

Shahed University

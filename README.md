# OCTess
Welcome to the repository for our research paper on automating the data extraction process of macular cube spectral domain optical coherence tomography (SD-OCT) data using optical character recognition (OCR) and deep learning. The algorithm we developed, named OCTess (portmanteau of OCT and Tesseract), is highly accurate, efficient, and a time-saving alternative to manual data extraction.

## Summary
In this study, we focused on developing an OCR algorithm, OCTess, to automatically extract clinical and demographic data from Cirrus SD-OCT macular cube reports. Our algorithm utilizes multiple models from Tesseract, an open-source OCR software library, and leverages pixel-based bounding box coordinates for each field of interest in the macular cube report. The extracted data is processed through a series of image processing operations to convert it to text.

OCTess extracts SD-OCT macular cube data with near-perfect and equivalent accuracy to a human while being significantly more efficient.

## Getting Started
To use OCTess, please follow these steps:
1. Clone this repository
2. Ensure you have the required dependencies installed, as listed in ```requirements.txt```
3. Move your Cirrus SD-OCT PDF/PNG files into the ```Input/``` directory. Alternatively, you can use the 5 example files that are already provided
4. Run the bash script ```./run.sh``` to execute the OCR algorithm and validate the results using the provided dataset

## Repository Structure
```Input/```: Input your raw SD-OCT macular cube reports in this directory. Delete the example files if you do not need them

```tessdata/```: Directory of saved Tesseract deep learning and legacy models

```patterns/```: Regex pattern rules used for data extraction

```pdf_to_img.py```: Python script to convert PDF files to PNG format (if they are not already PNG)

```extract_OCT.py```: Python script to extract data from each PNG file, organize it into a table and generate ```OCTess.xlsx```

```verify_OCT.py```: Python script that performs a series of verifications and highlights regions of ```OCTess.xlsx``` that may be erroneous

```requirements.txt```: Lists the necessary dependencies for this project


## Contributing
We welcome contributions to improve the algorithm or expand its applicability. Please feel free to submit issues, pull requests, or contact the authors directly.

## Author Contact:
Michael Balas: [michael.balas@mail.utoronto.ca](mailto:michael.balas@mail.utoronto.ca?subject=[GitHub]%20OCTess%20Inquiry)

## License
This project is licensed under the GNU GPLv3 License. See the ```LICENSE``` file for details.

## Citation
If you use this code or the results from our research paper in your work, please cite the paper using the following format:

```Citation Pending```

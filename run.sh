#!/bin/bash
PDF_PATH='./Input/'
PNG_PATH='./PNG/'
OUTPUT_PATH='./OCTess.xlsx'

python3 ./pdf_to_img.py --input $PDF_PATH --output $PNG_PATH --verbose True
python3 ./extract_OCT.py --input $PNG_PATH --save $OUTPUT_PATH --verbose True
python3 ./verify_OCT.py --input $OUTPUT_PATH --output $OUTPUT_PATH


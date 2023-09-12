import os
from pdf2image import convert_from_path
import optparse
import cv2

## Set Options Parser
parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", dest="ipath", default="./PDF/", help="Path to directory storing PDF files")
parser.add_option('-o', '--output', action="store", dest="opath", default="./PNG/", help="Path to directory to store PDF-->PNG files")
parser.add_option('-v', '--verbose', action="store", dest="verb", default="T", help="Print information to screen")

options, args = parser.parse_args()

INPUT_PATH = str(options.ipath)
INPUT_FILES = [os.path.join(INPUT_PATH, pdf_file) for pdf_file in os.listdir(INPUT_PATH)]
if os.path.join(INPUT_PATH, '.DS_Store') in INPUT_FILES: INPUT_FILES.remove(os.path.join(INPUT_PATH, '.DS_Store'))
OUTPUT_PATH = str(options.opath)
if not os.path.exists(OUTPUT_PATH):
   os.makedirs(OUTPUT_PATH)

VERBOSE = str(options.verb).lower() in ['t', 'true']

print('----------------------------------------')
print('Image Conversion Process (PDF --> PNG) Initializing...')
print('----------------------------------------')
for filename in sorted(os.listdir(INPUT_PATH)):
    file_path = os.path.join(INPUT_PATH, filename)
    if file_path.endswith('.pdf'):
        if VERBOSE:
            print('Converting File: ', filename)
        images = convert_from_path(file_path, dpi=500)
        new_filename = os.path.join(OUTPUT_PATH, filename[:-4] + '.png')
        images[0].save(new_filename, 'PNG')

print('----------------------------------------')
print('Image Conversion Process Complete.')
print('----------------------------------------')

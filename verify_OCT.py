import os
import pandas as pd
import numpy as np
import math
import optparse

## Set Options Parser
parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", dest="ipath", default="./OCTess.xlsx", help="Path to raw OCTess output.")
parser.add_option('-o', '--output', action="store", dest="opath", default="./OCTess.xlsx", help="File path to save verified OCTess output file (default settings automatically override raw file)")

options, args = parser.parse_args()

INPUT_FILE = str(options.ipath)
OUTPUT_FILE = str(options.opath)


def flag_numeric(val, pmean, thresh):
    try:
        val = float(val)
    except:
        color = '#FFCCCB'
        return 'background-color: {}'.format(color)
    if (math.isnan(val)):
        color = 'yellow'
    elif ((val > (pmean+thresh)) | (val < (pmean-thresh))):
        color = 'yellow'
    else:
        return False
    return 'background-color: {}'.format(color)

def flag_gender(val):
    if val.lower() not in {'male', 'female'}:
        color = '#FFCCCB'
    else:
        return False
    return 'background-color: {}'.format(color)
    
def flag_eye(val):
    if val.lower() not in {'od', 'os'}:
        color = '#FFCCCB'
    else:
        return False
    return 'background-color: {}'.format(color)

def flag_signal(val):
    try:
        val = int(val)
    except:
        color = '#FFCCCB'
        return 'background-color: {}'.format(color)
    if ((val > 10) | (val < 0)):
        color = '#FFCCCB'
    else:
        return False
    return 'background-color: {}'.format(color)

if __name__ == '__main__':
    print('Verifying Data...')
    df = pd.read_excel(INPUT_FILE, dtype='str')
    styler = df.copy().style
    styler.applymap(flag_gender, subset='Gender')
    styler.applymap(flag_eye, subset='Eye')
    styler.applymap(flag_signal, subset='Signal_Strength')
    numeric_params = ['Superior', 'Central_Superior', 'Nasal', 'Central_Nasal', 'Inferior',
    'Central_Inferior', 'Temporal', 'Central_Temporal', 'Central', 'Volume', 'Avg_Thickness']
    for param in numeric_params:
        df[param] = pd.to_numeric(df[param], errors='coerce')
        # Calculate the mean of the parameter of interest
        param_mean = df[param].mean()
        # Calculate the standard deviation of the parameter of interest
        param_std = df[param].std()
        # Calculate the threshold for values that are greater than three standard deviations
        threshold = 3 * param_std
        styler.applymap(flag_numeric, pmean=param_mean, thresh=threshold, subset=param)
    styler.to_excel(OUTPUT_FILE, index=False)
    print('Data Verification Complete.')

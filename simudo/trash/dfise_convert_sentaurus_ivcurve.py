'''
This exists because Sentaurus sometimes uses "DF-ISE" files for output.
'''

from argh import ArghParser, arg
import re
import shlex
import numpy as np
import pandas as pd

@arg('input_files', help="plt files", nargs='+')
def convert(input_files):
    for input_file in input_files:
        convert_single(input_file)

def convert_single(input_file):
    J_SCALE = 1e10

    with open(input_file, 'rt') as h:
        data = h.read()

    m = re.search(r"\s+datasets\s*=\s*\[(.+?)\]", data, flags=re.DOTALL)
    cols = shlex.split(m.group(1))

    m = re.search(r"Data\s*\{(.+?)\}", data, flags=re.DOTALL)
    data = [float(x) for x in shlex.split(m.group(1))]

    df = pd.DataFrame(data=np.array(data).reshape(-1, len(cols)),
                      columns=cols)
    dfo = pd.DataFrame({
        'Vext': df['anode InnerVoltage'] - df['cathode InnerVoltage']})
    for k in ['e', 'h', 'Total']:
        df['cathode {}Current'.format(k)] *= -1

    for band_before, bandk in [
            ('h', 'VB'), ('e', 'CB'), ('Total', 'tot')]:
        for contact_before, contact_k in [('anode', 'p'), ('cathode', 'n')]:
            dfo['j_{}:{}_contact'.format(bandk, contact_k)] = J_SCALE*(
                df['{} {}Current'.format(contact_before, band_before)])

    # clip to zero
    dfo['Vext'][dfo['Vext'].abs() < 1e-6] = 0.0

    SPLIT = True
    if SPLIT:
        for sgn in [1, -1]:
            xdf = dfo[dfo['Vext']*sgn >= 0].copy()
            xdf['_Vext_abs'] = dfo['Vext'].abs()
            xdf.sort_values('_Vext_abs', inplace=True)
            del xdf['_Vext_abs']
            xdf.reset_index(drop=True, inplace=True)
            xdf.to_csv(input_file + '_' + ' pm'[sgn]+'.csv')
    else:
        # print(dfo)
        dfo.to_csv(input_file+'.csv')

parser = ArghParser()
parser.add_commands([convert])

if __name__ == '__main__':
    parser.dispatch()




import os
import pandas as pd

def main():
    directory = "sentaurus-study2-iv"
    for f in os.listdir(directory):
        with open(os.path.join(directory, f), 'rt') as handle:
            df = pd.read_csv(handle, skiprows=[1])
        for k in df.columns:
            if str(k).startswith('j_'):
                df[k] *= 1e8 # mA/um^2 -> mA/cm^2
        df.to_csv(os.path.join(directory, f))

if __name__ == '__main__':
    main()




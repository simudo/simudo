
import os
import pandas as pd

def main():
    directory = "sentaurus-1e18-passivated-1d"
    for f in os.listdir(directory):
        with open(os.path.join(directory, f), 'rt') as handle:
            df = pd.read_csv(handle, skiprows=[1])
        df.sort_values('Y', inplace=True)
        mu_p = df['hQuasiFermiPotential'].values[0]
        mu_n = df['eQuasiFermiPotential'].values[-1]

        V_str = '{:g}'.format(round(mu_p - mu_n, 5))
        newname = 'diode_1d b=128 min_srv=0 c=1e18 V={}.csv'.format(V_str)
        print(newname)
        #os.rename(os.path.join(directory, f), os.path.join(directory, newname))

if __name__ == '__main__':
    main()

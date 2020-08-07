import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("output V_stepper_parameter.csv", skiprows=[1])
    df['V'] = df['sweep_parameter:parameter']
    df['J'] = df['avg:current_CB:p_contact'] + df['avg:current_VB:p_contact']
    df = df[df['V'] != 0]

    fig, ax = plt.subplots()
    ax.plot(df['V'], df['J'], label="$J(V)$", marker='o')
    ax.set_yscale('log')
    ax.set_xlabel(r'applied potential ($\mathrm{V}$)')
    ax.set_ylabel(r'total current ($\mathrm{mA}/\mathrm{cm}^2$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig("tutorial-pn-diode-JV.png")

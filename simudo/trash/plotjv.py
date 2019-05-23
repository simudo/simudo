from .. import pyplt as pplt

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

def plot_jv():
    df = pd.read_csv("out/jv.csv")
    with pplt.plot01("out/jv.png") as (fig, ax):
        ax.plot(df.V, df.j1, label="J")

plot_jv()

from matplotlib import pyplot as plt
import os
import numpy as np
from ..util.pdftoppm import savefig

def align_x_positions(labels):
    xs = [label.get_transform().transform(label.get_position())[0]
          for label in labels]

    new_x = min(xs)

    for label in labels:
        trans = label.get_transform()
        p = trans.transform(label.get_position())
        p[0] = new_x
        label.set_position(trans.inverted().transform(p))

def main():
    fig, ax = plt.subplots()

    Xs = np.linspace(0, 1, 100+1)
    Ys = np.sin(10 * Xs**3) * 30
    ax.plot(Xs, Ys)
    ax.set_xlabel("ecks label")
    ax.set_ylabel("why label")

    align_x_positions([ax.xaxis.label, ax.yaxis.label])

    savefig(fig, "out/hello.pdf")

    return locals()

if __name__ == '__main__':
    main()



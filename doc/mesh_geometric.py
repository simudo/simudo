
# see doc/mesh.lyx

from numpy import array, diff

def main():
    f = 1.3
    a = 1

    # geometrically expanding mesh
    X = [0]
    for k in range(1, 7):
        X.append(X[-1] + a * f**(k-1))

    X = array(X)
    D = diff(X)

    # this is very unintuitive, but correct
    D1 = (f-1) * X + a

    print(X)
    print(D)
    print(D1)

if __name__ == '__main__':
    main()


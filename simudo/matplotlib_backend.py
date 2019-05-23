
def get_pyplot():
    import matplotlib
    import os

    if os.environ.get('INTERACTIVE', 'n')[0:1].lower() != 'y':
        matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size':16})

    from matplotlib import pyplot

    return pyplot


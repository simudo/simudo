
import lzma
import yaml
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools
from simudo.util.pint import make_unit_registry
from simudo.misc import sentaurus_import

def main():
    U = make_unit_registry()
    def write(filename, tree):
        with open(filename, 'wt') as h:
            yaml.dump(tree, h)

    lst_mp = (
        -1,
        # 60,
        # 200,
        # 1000,
        # 2000,
    )
    lst_bchack = (0, 1)
    lst_ffill = (0, 1)
    lst_thresh = ( # fta, ftr, fwz
        # ('0', '1e-10', 0),
        # ('0', '1e-4', 0),

        # ('0', '0', 1),
        ('0', '0', 0),
        # ('1e-1', '0', 0),
        # ('inf', '0', 0),

        # ('1e-10', '0', 0), ('1e-5', '0', 0),
        # ('0', '1e-10', 0), ('0', '1e-5', 0),
    )
    lst_quaddeg = (
        # (30, 30, None),
        (20, 8, None),
    )
    lst_transport_method = (
        'mixed_qfl_j',
        'mixed_u_j',
    )

    FROM_IVCURVES = True
    ALL_IVCURVE_POINTS = False

    if FROM_IVCURVES:
        sentaurus_runs = defaultdict(list)
        sentaurus_device_lengths = {}
        for sentaurus_filename, d in \
          sentaurus_import.list_sentaurus_diode_1d_ivcurves():
            V_str = d['V']
            V = float(V_str)
            Vdir = 1 if V >= 0 else -1

            df = pd.read_csv(sentaurus_filename, index_col=0)

            previous_V = np.inf
            for index, row in df.to_dict('index').items():
                V = float(row['Vext'])

                if (not ALL_IVCURVE_POINTS) and (abs(previous_V - V) < 0.02):
                    continue

                sentaurus_runs[(d['c'], d['min_srv'], Vdir)].append({
                    'filename': None,
                    'ivcurve_filename': sentaurus_filename,
                    'ivcurve_row_index': index,
                    'Vext_str': '{:.5f}'.format(V),
                    'Vext': V,
                    'device_length': d['device_length']})

                previous_V = V
    else:
        sentaurus_runs = defaultdict(list)
        sentaurus_device_lengths = {}
        for sentaurus_filename, d in \
          sentaurus_import.list_sentaurus_diode_1d_data():
            V_str = d['V']
            V = float(V_str)
            Vdir = 1 if V >= 0 else -1

            with lzma.open(sentaurus_filename, mode='rb') as handle:
                Sdf, Sdf_units = sentaurus_import.read_df_unitful(
                    handle=handle,
                    unit_registry=U)

            sentaurus_runs[(d['c'], d['minority_contact'], Vdir)].append({
                'filename': sentaurus_filename,
                'Vext_str': V_str,
                'Vext': V,
                'device_length': float(Sdf['x'].values[-1])})

    for l in sentaurus_runs.values():
        l.sort(key=lambda d: abs(d['Vext']))

    for sentaurus_run_key, sentaurus_run in sentaurus_runs.items():

        doping, minority_srv, V_ext_direction = sentaurus_run_key
        device_length = next(iter(sentaurus_run))['device_length']

        for mp, tmethod, bchack, ffill, (fta, ftr, fwz), (
                qd1, qd2, qdr) in itertools.product(
                    lst_mp, lst_transport_method, lst_bchack,
                    lst_ffill, lst_thresh, lst_quaddeg):
            if not (not bchack and ffill): continue

            # if doping not in ('1e18', '1e20'): continue
            if doping not in ('1e18',): continue

            if tmethod == 'mixed_u_j' and (fta, ftr, fwz) != ('0', '0', 0):
                continue

            if not ffill and not bchack: # sanity
                continue
            title = "bc20191117 c={} min_srv={} Vd={} meshp={} M={} bchack={} ffill={} fta={} ftr={} fwz={} qd1={} qd2={} qdr={}".format(
                doping, minority_srv, V_ext_direction,
                mp, tmethod, bchack, ffill,
                fta, ftr, fwz,
                qd1, qd2, qdr)

            # if title != 'bc20191117 c=1e18 Vd=1 meshp=-1 bchack=0 ffill=1 fta=0 ftr=0 fwz=0 qd1=30 qd2=30 qdr=None':
            #     continue

            # if title != 'bc20191117 c=1e18 Vd=1 meshp=200 bchack=0 ffill=1 fta=0 ftr=1e-10 fwz=0':
            #     continue

            o = "out/" + title
            os.makedirs(o, exist_ok=True)

            write(o+'/submit.yaml',
                  dict(parameters=dict(
                      device_length=device_length,
                      doping=doping,
                      min_srv=minority_srv,
                      bchack=bchack,
                      fill_from_boundary=ffill,
                      mesh_points=mp,
                      transport_method=tmethod,
                      title=title,
                      fill_threshold_abs=fta,
                      fill_threshold_rel=ftr,
                      fill_with_zero_except_bc=fwz,
                      quaddeg_super=qd1,
                      quaddeg_g=qd2,
                      quaddeg_rho=qdr,
                      voltage_steps=sentaurus_run)))

if __name__ == '__main__':
    main()


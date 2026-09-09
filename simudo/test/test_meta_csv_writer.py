"""``MetaCSVWriter`` must append rows in the order of the header already in
the file, whatever the order of the new process's ``meta`` dictionary.

A resumed run (or the negative-voltage branch of a sweep) appends to the
``sim_V.csv`` written by an earlier process; the column order of ``meta``
came from set iteration, which differs between processes, so without this the
appended rows had, e.g., the left-contact current under the right-contact
header.

Created by Claude Fable 5.1 (Anthropic), 2026-09-04.
"""

import csv

from ..io.output_writer import MetaCSVWriter


def _meta(order, values):
    return {k: dict(value=values[k], units='mA/cm^2') for k in order}


def test_append_follows_existing_header(tmp_path):
    fn = str(tmp_path / 'sim_V.csv')
    cols = ['sweep_parameter:V', 'J:right', 'J:left']
    w = MetaCSVWriter(fn, _meta(cols, {'sweep_parameter:V': 0.8, 'J:right': 1.0, 'J:left': 2.0}))
    w.add_row(_meta(cols, {'sweep_parameter:V': 0.8, 'J:right': 1.0, 'J:left': 2.0}))
    w.file.close()

    # a later process registers the same columns in a different order
    cols2 = ['sweep_parameter:V', 'J:left', 'J:right']
    w2 = MetaCSVWriter(fn, _meta(cols2, {'sweep_parameter:V': 0.9, 'J:left': 20.0, 'J:right': 10.0}))
    w2.add_row(_meta(cols2, {'sweep_parameter:V': 0.9, 'J:left': 20.0, 'J:right': 10.0}))
    w2.file.close()

    with open(fn, newline='') as fh:
        rows = [r for r in csv.reader(fh) if not r[0].startswith('#')]
    header = rows[0]
    assert header == cols
    data = {float(r[0]): dict(zip(header, map(float, r))) for r in rows[1:]}
    assert data[0.8] == {'sweep_parameter:V': 0.8, 'J:right': 1.0, 'J:left': 2.0}
    assert data[0.9] == {'sweep_parameter:V': 0.9, 'J:right': 10.0, 'J:left': 20.0}

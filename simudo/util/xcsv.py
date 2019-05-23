'''
This (standalone) module implements a Pandas CSV reader-writer pair
that allows data types to survive a round-trip (where they wouldn't
using plain pandas ``to_csv``). It achieves this by saving some column
metadata to JSON, and by prefixing string values with a ":" character
so that they cannot be confused with NaN values (which are also
allowed in string columns, creating unresolvable ambiguity in the
written data).

See :py:meth:`~XCSV.to_csv` and :py:meth:`~XCSV.read_csv` for more info.

These methods are available as simple functions, so you can do::

    >>> to_xcsv(df, "hello.csv")
    >>> df2, meta = from_xcsv(df, "hello.csv")
'''

import functools
import json
import os
import re
import unittest

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from cached_property import cached_property

__all__ = ['to_xcsv', 'read_xcsv',
           'XCSV', 'XCSVBase', 'XCSVWriter', 'XCSVReader']


class XCSVBase():
    _STRING_PREFIX = ':'
    _XCSV_VERSION = 1

    def json_path_from_xcsv_path(self, path):
        return path + '_meta.json'

    @cached_property
    def json_path(self):
        return self.json_path_from_xcsv_path(self.path)

    def is_string_column(self, df, dtypes_dict, column_name):
        return dtypes_dict[column_name] == 'object'

class XCSVWriter(XCSVBase):
    @property
    def string_prefix(self):
        return self._STRING_PREFIX

    @cached_property
    def dtypes_dict(self):
        df = self.df
        ddf = df.dtypes.to_frame('dtypes').reset_index()
        dtypes_dict = ddf.set_index('index')['dtypes'].astype(str).to_dict()
        return dtypes_dict

    def compute_meta(self):
        return {
            'columns': list(self.df.columns),
            'dtypes': self.dtypes_dict,
            'string': list(sorted(self.string_columns)),
            'string_prefix': self._STRING_PREFIX,
            'xcsv_version': self._XCSV_VERSION}

    @cached_property
    def meta(self):
        return self.compute_meta()

    @cached_property
    def string_columns(self):
        dtypes_dict = self.dtypes_dict
        df = self.df
        return set(k for k in dtypes_dict.keys()
                   if self.is_string_column(df, dtypes_dict, k))

    def to_csv(self):
        df = self.df
        dtypes_dict = self.dtypes_dict

        _STRING_PREFIX = self._STRING_PREFIX
        def string_func(value):
            if isinstance(value, str):
                return _STRING_PREFIX + value
            else:
                return value

        dfc = pd.DataFrame(index=df.index)
        for k in df.columns:
            series = df[k]
            if k in self.string_columns:
                series = series.map(string_func)
            dfc[k] = series

        self.write_json_meta()
        dfc.to_csv(self.path, **self.to_csv_kwargs)

    def write_json_meta(self):
        with open(self.json_path, 'wt') as h:
            json.dump(self.meta, h, sort_keys=True, indent=2)

class XCSVReader(XCSVBase):
    def load_json_meta(self):
        with open(self.json_path, 'rt') as h:
            return json.load(h)

    @cached_property
    def meta(self):
        return self.load_json_meta()

    @cached_property
    def dtypes_dict(self):
        return self.meta['dtypes']

    @cached_property
    def string_columns(self):
        return set(self.meta['string'])

    @cached_property
    def string_prefix(self):
        return self.meta['string_prefix']

    def read_csv(self):
        meta = self.meta

        xcsv_ver = meta['xcsv_version']
        if xcsv_ver != self._XCSV_VERSION:
            raise ValueError("unexpected xcsv_version {!r} (expected {!r})"
                             .format(xcsv_ver, self._XCSV_VERSION))

        df = pd.read_csv(
            self.path, dtype=self.dtypes_dict,
            **self.read_csv_kwargs)

        dfc = pd.DataFrame(index=df.index)
        for k in df.columns:
            series = df[k]
            if k in self.string_columns:
                self.process_string_column(
                    df=df, column_name=k, series=series)
            dfc[k] = series

        return (dfc, meta)

    @cached_property
    def string_prefix_re(self):
        return '^' + re.escape(self.string_prefix)

    def process_string_column(self, df, column_name, series):
        series.replace(
            {'True': True, 'False': False}, inplace=True)
        # TODO: parse numerical values
        series.replace(
            self.string_prefix_re,
            '', regex=True, inplace=True)

class XCSV():
    reader_class = XCSVReader
    writer_class = XCSVWriter

    @classmethod
    def to_csv(cls, df, path, json_path=None, to_csv_kwargs={}):
        '''Basically the same as :py:meth:`pandas.DataFrame.to_csv`, but
with proper escaping for strings to prevent them from being
accidentally parsed as numbers or nan, and with column dtypes being
written to an accompanying json file.

If the csv filename is :code:`"a.csv"`, then the file name containing the
metadata will be called :code:`"a.csv_meta.json"`.

"XCSV" pronounced "excessive".

Warning: mixed-type ("object") columns are assumed to be string
columns. So make sure those don't contain anything other than strings
or NaN, or your else your data might not survive the roundtrip test.

What's definitely safe:

- Columns with floats/ints and nans.
- Columns with strings and nans.
- Columns with booleans (no nans allowed!).
'''
        obj = cls.writer_class()
        obj.df = df
        obj.path = path
        if json_path is not None:
            obj.json_path = json_path
        obj.to_csv_kwargs = to_csv_kwargs
        return obj.to_csv()

    @classmethod
    def read_csv(cls, path, json_path=None, read_csv_kwargs={}):
        '''Opposite of :py:meth:`to_csv`.'''

        obj = cls.reader_class()
        obj.path = path
        if json_path is not None:
            obj.json_path = json_path
        obj.read_csv_kwargs = read_csv_kwargs
        return obj.read_csv()

to_xcsv = XCSV.to_csv
read_xcsv = XCSV.read_csv


class TestMe(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a = pd.DataFrame({
                'ints': [1,2,3,4],
                'ints_and_nan': [np.nan,2,3,4],
                'floats': [np.nan, 0.2, 0.3, 0.4],
                'strings': "goodbye cruel world xoxo".split(),
                'strings_and_nan': [np.nan, "NaN", "2", "4"],
                'bool': [True, False, False, True],
                # 'bool_and_nan': [True, np.nan, False, True], # not supported
                # 'mixed': [True, 'True', 2, 4.5] # not supported
            })
            # tmp = "/tmp"
            path = os.path.join(tmp, "example.csv")

            # print(a)
            # print(a.dtypes)
            to_xcsv(a, path)
            # print(open(path, 'rt').read())
            b = read_xcsv(path, read_csv_kwargs=dict(index_col=0))[0]

            # print(b)

            for df in [a, b]:
                # self.assertTrue(isinstance(df.a.iloc[0], int))
                self.assertTrue(isinstance(df.floats.iloc[0], float))
                self.assertTrue(isinstance(df.strings_and_nan.iloc[1], str))
                # self.assertTrue(isinstance(df.d.iloc[0], bool))

            N = len(a)
            for col in a.columns:
                for i in range(N):
                    va, vb = a[col].iloc[i], b[col].iloc[i]
                    try:
                        np.testing.assert_array_equal(va, vb)
                    except:
                        print("col={!r} i={} {!r} {!r}".format(col, i, va, vb))
                        raise


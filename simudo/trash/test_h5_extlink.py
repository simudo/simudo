import os
import h5py

print(h5py.version.version)
print(h5py.version.hdf5_version)
print(h5py.version.api_version)

f = h5py.File('ref.hdf5', 'w')
f.close()

r = h5py.File('test.rtin', 'w')
r['dust_001'] = h5py.ExternalLink('ref.hdf5', '/')
x = r['dust_001']

r2 = h5py.File('test2.rtin', 'w')
r2['dust_001'] = h5py.ExternalLink('ref.hdf5', '/')
y = r2['dust_001']

print(x, y)

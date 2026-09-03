"""Composition-dependent optical properties for alloys.

NOT PART OF THE RELEASED PACKAGE.  Simudo 0.7 ships measured optical data for
the binaries only.  This module and its ``alloy_optics/`` tables live in the
source tree but are deliberately excluded from the wheel and sdist (see
``package_data`` in setup.py), and nothing in the library calls them: the two
alloy materials use a flat, composition-independent ``opt_cv/alpha`` instead.

Kept for future work.  Two things need fixing before it can be wired up:
the data path is built from ``Path.cwd()`` rather than the package directory,
and ``optical_properties_table`` subscripts the 0-d array a
``RegularGridInterpolator`` returns for a single point.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pint import UnitRegistry, Quantity, UnitStrippedWarning
from pathlib import Path
import pandas, os
import dolfin, ufl
from warnings import simplefilter

"""
Interpolator for composition-dependent optical properties of alloys. Constructs tables like those served by `optical_properties_table` in non-alloys.

All AlGaAs data used here was collected in:
D. E. Aspnes, S. M. Kelso, R. A. Logan, R. Bhat. "Optical properties of AlxGa1-xAs". J. Appl. Phys. 60, 754-767 (1986) https://doi.org/10.1063/1.337426
The data has been collated and turned into .csv files by:
https://refractiveindex.info/?shelf=other&book=AlAs-GaAs&page=Aspnes-0
That database is released under CC0 1.0, and asks to be cited as:
M. N. Polyanskiy. "Refractiveindex.info database of optical constants".
Sci. Data 11, 94 (2024). https://doi.org/10.1038/s41597-023-02898-2

All GaNAs data used here was collected in:
G. Leibiger, V. Gottschalch, B. Rheinländer, J. Šik, M. Schubert
"Model dielectric function spectra of GaAsN for far-infrared and near-infrared to ultraviolet wavelengths".
J. Appl. Phys. 1 May 2001; 89 (9): 4927–4938. https://doi.org/10.1063/1.1359422
"""

material_names = [
	["algaas", "aluminumgalliumarsenide"],
	["ganas", "gaasn", "galliumarsenidenitride", "galliumnitridearsenide"],
	]

simplefilter(action='ignore', category=UnitStrippedWarning)
class OpticsInterpolator():

	__all__ = ["optical_properties_table"]
	U = UnitRegistry()

	def __init__(self, material: str):
		r'''
		Provides methods for getting the abosorbance and refractive index of a material.

		Parameters
		----------

		material: str
			The name of the material. Takes any usage of upper or lower case spelling.
		'''
		materialIndex = -1
		for i in range(len(material_names)):
			if (material.lower() in material_names[i]):
				materialIndex = i
				break
		self.alloyData = str(Path.cwd()) + "/alloy_optics/"

		match (materialIndex):
			case -1:
				raise NotImplementedError("OpticsInterpolator does not currently implement your material: "+str(material))
			case 0:
				self.alloyData += "AlGaAs/"
			case 1:
				self.alloyData += "GaNAs/"
			case _:
				raise RuntimeError("Failed to find the index for your material: "+str(material))

		self.initialize_data()

	wavelengths = None
	fractions = []
	grid_values = dict({}) # dict of arrays with the same dimension as grid_points

	def initialize_data(self):
		'''
		Reads in optical data in a format well-suited for `scipy.interpolate.RegularGridInterpolator`.
		Requires alloy data to be named in the exact format `A_x B_1-x C` or `A B_x C_1-x`. Currently does not support quaternary alloys.
		'''
		self.fractions.clear()
		data = dict({})
		ls = os.listdir(path = self.alloyData)
		ls.sort()
		for filename in ls:
			if filename.endswith(".csv"):
				df = pandas.read_csv(self.alloyData+filename)
				wl = df["wl"].to_numpy()
				elements = filename[:-4].split(' ')
				e = elements[0].split('_')
				if len(e) > 1: alloy_fraction = float(e[1])
				else: alloy_fraction = float(elements[1].split('_')[1])

				data[alloy_fraction] = df # Store all the data temporarily so as to not need to crawl through a second time 
				if (self.wavelengths is None):
					self.wavelengths = wl
				self.fractions.append(alloy_fraction)

		# Alloy wavelengths are saved in um while non-alloys saved in nm
		self.wavelengths *= 1000
		self.fractions = np.array(self.fractions)

		wlL = len(self.wavelengths); frL = len(self.fractions)
		self.grid_values = {
			"a": np.zeros((wlL, frL)),
			"n": np.zeros((wlL, frL)),
			"k": np.zeros((wlL, frL)),
			}
		
		for col, frac in enumerate(data.keys()):
			df = data[frac]
			wl = df["wl"].to_numpy().astype(float)
			n = df["n"].to_numpy().astype(float)
			k = df["k"].to_numpy().astype(float)

			for row, w in enumerate(wl):
				self.grid_values["n"][row, col] = n[row]
				self.grid_values["k"][row, col] = k[row]
				self.grid_values["a"][row, col] = 4*np.pi*k[row]/w * 10**4

		self.rgi = {}
		for v in ('a', 'n', 'k'):
			self.rgi[v] = RegularGridInterpolator(points=(self.wavelengths, self.fractions), values = self.grid_values[v])

	def optical_properties_table(self, alloy_fraction:float):
		'''Returns a table of optical properties identical in form to those used in non-alloy materials.'''
		if alloy_fraction < self.fractions[0] or alloy_fraction > self.fractions[-1]:
			# TODO: This should instead do a linear extrapolation from the two highest/lowest alloy fraction tables.
			# print(f"For {self.alloyData.split('/')[-2]}, {alloy_fraction} is outside the range of available data ({self.fractions[0]}, {self.fractions[-1]}). Extrapolation may be inaccurate.")
			raise RuntimeError(f"For {self.alloyData.split('/')[-2]}, {alloy_fraction} is outside the range of available data ({self.fractions[0]}, {self.fractions[-1]})")
		new_table = [
			(
				wl,
				self.rgi['a']((wl, alloy_fraction))[0],
				self.rgi['n']((wl, alloy_fraction))[0],
				self.rgi['k']((wl, alloy_fraction))[0]
			) for wl in self.wavelengths
		]

		return np.array(new_table)
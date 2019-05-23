
import numpy as np
from numpy import nan

__all__ = ['PygmshMakeRegions']

# FIXME: clean this up

class PygmshMakeRegions(object):
    '''We want overlapping regions. gmsh and pygmsh make that
difficult. gmsh supports overlapping physical surfaces/volumes, but it
returns duplicate mesh cells (see `meshio issue 175`_). We can fix
this by postprocessing pygmsh's output, which thankfully contains all
the necessarily info. *At least for now.*

.. _`meshio issue 175`: https://github.com/nschloe/meshio/issues/175
'''
    def process(self, cells, cell_data, field_data, dim):
        '''Note: In-place modifies ``cells`` and ``cell_data``.'''

        cells_key, = cells.keys()

        xcell_data = cell_data[cells_key]

        xcell_data_key = 'gmsh:physical'
        for k in list(xcell_data.keys()):
            if k != xcell_data_key:
                del xcell_data[k]

        r = self.unduplicate_msh3_cells(
            cells[cells_key],
            xcell_data[xcell_data_key])

        cells[cells_key] = r['cells']
        xcell_data[xcell_data_key] = r['cell_data']

        tagid_to_tagname = {tagid: tagname for tagname, (tagid, dimension)
                            in field_data.items()
                            if dimension == dim}

        tag_to_cell_values = {
            tagid_to_tagname[k]: v
            for k, v in r['tag_to_cell_values'].items()}

        return dict(tag_to_cell_values=tag_to_cell_values)

    def unduplicate_msh3_cells(self, cells, cell_data):

        previous_row = None
        new_cells_rowindices = []
        cell_to_tags = []
        last_cell_to_tags = None
        for old_index, row in enumerate(cells):
            if previous_row is not None and np.array_equal(row, previous_row):
                pass # duplicate cell
            else:
                previous_row = row
                new_cells_rowindices.append(old_index)
                last_cell_to_tags = set()
                cell_to_tags.append(last_cell_to_tags)
            last_cell_to_tags.add(cell_data[old_index])

        new_cells = cells[new_cells_rowindices]

        ### alternate code
        #
        # the code below messes up the cell ordering (because
        # np.unique sorts the array), and is quite inefficient because
        # gmsh seems to produce contiguous blocks of duplicate cells.
        #
        ### BEGIN OLD CODE
        #
        # new_cells, unique_inverse = np.unique(
        # cells, axis=0, return_inverse=True)
        #
        # cell_to_tags = [set() for i in range(len(new_cells))]
        # for old_index, new_index in enumerate(unique_inverse):
        #     cell_to_tags[new_index].add(cell_data[old_index])
        #
        ### END OLD CODE

        cell_to_tagtuple = [tuple(sorted(v)) for v in cell_to_tags]
        tagtuples = list(sorted(frozenset(cell_to_tagtuple)))
        tagtuple_to_cellvalue = {
            k: i for i, k in enumerate(tagtuples, 1)}

        tag_to_cell_values = {}
        for tt, cv in tagtuple_to_cellvalue.items():
            for tag in tt:
                cvs = tag_to_cell_values.get(tag, None)
                if cvs is None:
                    tag_to_cell_values[tag] = cvs = set()
                cvs.add(cv)

        cell_to_cellvalue = [
            tagtuple_to_cellvalue[tt] for tt in cell_to_tagtuple]

        new_cell_data = np.array(cell_to_cellvalue, dtype='uintp')

        return dict(cells=new_cells,
                    cell_data=new_cell_data,
                    tag_to_cell_values=tag_to_cell_values)

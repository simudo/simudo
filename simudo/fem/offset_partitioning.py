
import collections

import numpy as np
from cached_property import cached_property

import dolfin

from ..util import SetattrInitMixin

# example
# 100 101 102 101 50 30 10 11 12 11
# [100 101 102 101] [50 30] [10 11 12 11]

__all__ = ['PartitionOffsets']

class PartitionOffsets(SetattrInitMixin):
    '''Offset partitioner.

Attributes
----------
function: :py:class:`dolfin.Function`
    Input function. Must be DG0 function.
out_function: :py:class:`dolfin.Function`
    Output function. If not set, use :py:attr:`function`. Must be DG0
    function.
thresholds: sequence
    Thresholds, as list of tuples
    :code:`(absolute_threshold, relative_threshold)`.
mesh_util: :py:class:`.mesh_util.MeshUtil`
    Mesh utilities and data.
bc_cells:
    Pair `(cells, cellavgs)`, where both are equally sized arrays. The
    first array is the indices of BC-adjacent cells, and the second
    array are their values.

Notes
-----

The algorithm is as follows:

0. All cells are unmarked (have a marker value of nan), and
   unseen. The cell queue is empty.

1. The cells adjacent to boundary conditions are placed in the cell
   queue. They are marked with the boundary condition value.

2. While there exist unseen cells:

   3. If the cell queue is empty, pick one unseen cell and place it
      in the cell queue.

   4. Remove cell out of queue, and visit it:

      5. Consider the current cell as seen.

      6. If the cell is already marked, set ``this_marker`` to the marker
         value. Otherwise, set ``this_marker`` to the current cell value.

      7. For every unmarked and unseen neighbor: if its value is
         "close" to our marker value, mark it with ``this_marker``,
         and consider it "seen".

      8. Add newly-marked cells to the cell queue.

      9. If the current cell was unmarked *and* step 7 marked any
         cells, mark the current cell with ``this_marker``.

'''
    thresholds = ((1e-8, 0.0),)
    bc_cells = ((), ())
    debug_fill_with_zero_except_bc = False

    @cached_property
    def out_function(self):
        return dolfin.Function(self.mesh_util.space.DG0)

    def update(self):
        # TODO: handle more than 1 threshold
        (abs_threshold, rel_threshold), = tuple(
            sorted(self.thresholds))

        input_function = self.function
        out_function = self.out_function
        DG0 = input_function.function_space()
        mesh = DG0.mesh()
        D = mesh.topology().dim()
        mesh.init(D, D)
        nan = np.nan
        isnan = np.isnan

        vec = input_function.vector()
        N = len(vec)

        assert N == mesh.num_cells()

        # TODO: check this invariant
        # assert all(cell.index == dofmap_cell_dofs(cell.index)[0]
        #            for cell in dolfin.cells(mesh))

        if rel_threshold == 0 and abs_threshold == 0: # optimization
            if input_function is not out_function:
                out_function.vector()[:] = input_function.vector()[:]

            # apply BC on adjacent cells, and we're done
            bc_idx, bc_vals = self.bc_cells
            out_function.vector()[bc_idx] = bc_vals
        else:
            result = np.full(N, nan, dtype=np.float64)

            # adjacency
            cell_to_cells = (self.mesh_util.mesh_data
                             .cell_to_cells_adjacency_via_facet)

            seen_cells = np.zeros(N, dtype='bool')
            seen_cells_index = 0
            q = collections.deque()

            for cell_index, cell_value in zip(*self.bc_cells):
                result[cell_index] = cell_value
                q.appendleft(cell_index)

            if self.debug_fill_with_zero_except_bc:
                result[isnan(result)] = 0.0
            else:
                while True:
                    if not len(q): # step 3 true branch
                        # step 3, 4
                        while (seen_cells_index < N and
                               seen_cells[seen_cells_index]):
                            seen_cells_index += 1
                        if seen_cells_index >= N:
                            break # we're done, no unseen cells remain
                        cell = seen_cells_index
                    else:
                        cell = q.pop() # step 4

                    seen_cells[cell] = 1 # step 5

                    # step 6, bookkeeping for step 9
                    this_marker = result[cell]
                    was_unmarked = isnan(this_marker)
                    if was_unmarked:
                        this_marker = vec[cell]

                    has_marked_adj_cells = False
                    for adj in cell_to_cells[cell]:
                        if seen_cells[adj]:
                            continue # don't bother with already-seen cells
                        adj_val = vec[adj]
                        is_close = (abs(this_marker - adj_val) <=
                                    abs_threshold +
                                    rel_threshold*np.maximum(abs(this_marker),
                                                             abs(adj_val)))
                        if is_close: # step 7
                            result[adj] = this_marker
                            seen_cells[adj] = 1 # step 8
                            q.appendleft(adj)
                            has_marked_adj_cells = True

                    if has_marked_adj_cells and was_unmarked:
                        result[cell] = this_marker # step 9


            #unmarked = vec if unmarked_use_original else out_function.vector()
            unmarked = vec

            out_function.vector()[:] = np.where(
                isnan(result), unmarked, result)

        return out_function

def offsets_to_facetfunction(function, out_ff=None):
    '''
Parameters
----------
function : dolfin.Function
    Input DG0 function.

Returns
-------
ff : dolfin.MeshFunction
    Facet function: equal to 1 if facet separates two cells with
    different `function` values, and 0 otherwise.
'''
    space = function.function_space()
    mesh = space.mesh()
    D = mesh.geometry().dim()
    vec = function.vector()

    ff = out_ff
    if not ff:
        ff = dolfin.MeshFunction("int", mesh, D-1, 0)

    ffv = ff.array()
    for facet in dolfin.facets(mesh):
        l = facet.entities(D)
        if len(l) == 2:
            a, b = l
            if vec[a] != vec[b]:
                ffv[facet.index()] = 1

    return ff

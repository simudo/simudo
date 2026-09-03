import meshio
import numpy as np


def keep_only_largest_dimension_cells(mesh: meshio.Mesh) -> None:
    """
    Discard cells of lesser dimension. This is necessary because older dolfin
    versions cannot read XDMF files with such cells.
    """
    largest_dim_entity_index = max(
        enumerate(mesh.cells), key=lambda i_cell: i_cell[1].dim
    )[0]
    mesh.cells[:] = [mesh.cells[largest_dim_entity_index]]
    for k, v in mesh.cell_sets.items():
        v[:] = [v[largest_dim_entity_index]]


def cell_sets_to_cell_values(
    mesh: meshio.Mesh,
) -> dict[str, dict[str, frozenset[int]]]:
    """
    In newer pygmsh versions, physical groups become meshio cell_sets. This
    function converts the cell_sets data to the usual integer cell values
    that are used inside Simudo. This modifies in-place the mesh object.
    """

    # NOTE: This function could actually theoretically work with multiple cell
    # types (eg line and triangle and point and tetrahedron). However dolfin
    # is limited in reading such meshes.

    cell_sets = mesh.cell_sets_dict

    entity_to_count = {k: len(v) for k, v in mesh.cells_dict.items()}

    physicals = list(cell_sets)
    physical_to_index = {k: i for i, k in enumerate(physicals)}

    # matrix where rows are physical groups and columns are entities
    entity_to_matrix = {
        entity: np.zeros((len(physicals), count), dtype="bool")
        for entity, count in entity_to_count.items()
    }

    for physical, d in cell_sets.items():
        i = physical_to_index[physical]
        for entity, entity_ids in d.items():
            entity_to_matrix[entity][i, entity_ids] = True

    entity_to_cell_function = {}
    entity_to_tag_to_cell_values = {}
    current_cell_value = 1
    for entity, matrix in entity_to_matrix.items():
        matrix_rows = sorted(frozenset(tuple(row) for row in matrix.T))
        matrix_row_tuple_to_cell_value_and_tags = {
            row: (
                cell_value,
                frozenset(
                    t for t, indicator in zip(physicals, row) if indicator
                ),
            )
            for cell_value, row in enumerate(matrix_rows, current_cell_value)
        }
        matrix_row_to_cell_value = {
            row: cell_value
            for row, (
                cell_value,
                _,
            ) in matrix_row_tuple_to_cell_value_and_tags.items()
        }
        current_cell_value += len(matrix_rows)

        entity_to_cell_function[entity] = np.array(
            [matrix_row_to_cell_value[tuple(row)] for row in matrix.T],
            dtype="int64",
        )

        entity_to_tag_to_cell_values[entity] = tag_to_cell_values = {
            tag: set() for tag in physicals
        }
        for (
            cell_value,
            tags,
        ) in matrix_row_tuple_to_cell_value_and_tags.values():
            for tag in tags:
                tag_to_cell_values[tag].add(cell_value)

    # remove cell sets and replace with cell values
    mesh.cell_sets_dict.clear()
    mesh.cell_sets.clear()
    mesh.cell_data.clear()
    mesh.cell_data["cv"] = cell_data_values = []

    for mesh_block in mesh.cells:
        cf = entity_to_cell_function.get(mesh_block.type, None)
        cell_data_values.append(cf)

    return entity_to_tag_to_cell_values


def strip_z_component(mesh: meshio.Mesh):
    """
    Remove z coordinate component to make it a 2d mesh.
    """
    mesh.points = mesh.points[..., :2]

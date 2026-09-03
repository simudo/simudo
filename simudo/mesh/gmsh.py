from collections import defaultdict
from collections.abc import Iterable
import itertools
from pathlib import Path
import tempfile
from typing import Optional
from weakref import WeakSet

import attr
import dolfin
import gmsh
import numpy as np

from . import meshio as simudo_meshio


@attr.s(frozen=True, slots=True)
class GmshTag:
    """
    GMSH internal tags.

    These change all the time if you do Boolean operations, do not
    rely on them!! Use TagsBase/TagsRef instead! Or rather, just use
    :meth:`TagTracker.ref`!
    """

    dim: int = attr.ib()
    tag: int = attr.ib()

    @property
    def dim_tag(self):
        return (self.dim, self.tag)

    @property
    def _id(self):
        # for pygmsh compatibility
        return self.tag


def _merge_binary_ops(cls, a, b):
    a_is_cls = isinstance(a, cls)
    b_is_cls = isinstance(b, cls)

    if a_is_cls and b_is_cls:
        return cls(a.operands + b.operands)

    if a_is_cls:
        return cls(a.operands + (b,))

    if b_is_cls:
        return cls((a,) + b.operands)

    return cls((a, b))


class TagsBase:
    """
    Base class for a collection of tags that is typically managed by
    a :class:`TagTracker`.
    """

    def get_gmsh_tags(self) -> Iterable[GmshTag]:
        raise NotImplementedError

    def get_gmsh_tag_list(self) -> list[tuple[int, int]]:
        return [x.dim_tag for x in self.get_gmsh_tags()]

    def get_pygmsh_tag_list(self) -> list[tuple[int, int]]:
        from pygmsh.common.dummy import Dummy  # optional dependency

        return [Dummy(*x.dim_tag) for x in self.get_gmsh_tags()]

    def __iter__(self):
        return iter(self.get_gmsh_tags())

    def __and__(self, other):
        if isinstance(other, TagsBase):
            return _merge_binary_ops(TagsIntersection, self, other)

        return NotImplemented

    def __or__(self, other):
        if isinstance(other, TagsBase):
            return _merge_binary_ops(TagsUnion, self, other)

        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TagsBase):
            return TagsDifference(self, other)

        return NotImplemented


@attr.s(eq=False, hash=False)
class TagsRef(TagsBase):
    tag_children: "set[GmshTag]" = attr.ib()

    def get_gmsh_tags(self):
        return self.tag_children


@attr.s
class TagsIntersection(TagsBase):
    operands: tuple[TagsBase] = attr.ib(converter=tuple)

    def get_gmsh_tags(self):
        if not self.operands:
            return set()

        ops = iter(self.operands)
        values = set(next(ops).get_gmsh_tags())
        for o in ops:
            if not values:
                break  # empty set

            values &= o.get_gmsh_tags()

        return values


@attr.s
class TagsUnion(TagsBase):
    operands: tuple[TagsBase] = attr.ib(converter=tuple)

    def get_gmsh_tags(self):
        s = set()
        for o in self.operands:
            s.update(o.get_gmsh_tags())
        return s


@attr.s
class TagsDifference(TagsBase):
    a: TagsBase = attr.ib()
    b: TagsBase = attr.ib()

    def get_gmsh_tags(self):
        s = set(self.a)
        s.difference_update(self.b)
        return s


@attr.s(eq=False)
class TagTracker:
    """
    Keeps track of gmsh tags as they change or multiply due to boolean
    operations.
    """

    tag_to_ref: dict[GmshTag, set[TagsRef]] = attr.ib(
        factory=lambda: defaultdict(WeakSet), init=False
    )
    env = attr.ib(default=gmsh.model.occ, init=False)

    @staticmethod
    def _ensure_gmsh_tag(x):
        if not isinstance(x, GmshTag):
            return GmshTag(*x.dim_tag)
        else:
            return x

    def ref(self, gmsh_tags: Iterable[GmshTag]) -> TagsRef:
        """
        Create a reference to one or more GmshTags. The reference remains
        valid even if the tags are modified or replaced by boolean operations.
        """
        if not isinstance(gmsh_tags, Iterable):
            gmsh_tags = [gmsh_tags]

        # TODO: is it worth deduplicating TagRef objects?
        f = self._ensure_gmsh_tag
        ref = TagsRef(set(f(x) for x in gmsh_tags))
        for tag in ref.get_gmsh_tags():
            self.tag_to_ref[tag].add(ref)
        return ref

    def __call__(self, *args, **kwargs):
        return self.ref(*args, **kwargs)

    def _replace(self, replacements: dict[GmshTag, set[GmshTag]]):
        """
        Modify all TagsRef objects to reflect the replacement of old GmshTag
        objects with newly generated ones.
        """
        tag_to_ref = self.tag_to_ref
        later = []

        for old, new_tags in replacements.items():
            old_refs = tag_to_ref.pop(old, None)
            if old_refs is None:
                continue

            while True:
                # stupid paranoid way to iterate the weak set while also
                # removing the items
                try:
                    ref = old_refs.pop()
                except KeyError:
                    break

                ref.tag_children.remove(old)
                later.append((ref, new_tags))

        for ref, new_tags in later:
            ref.tag_children.update(new_tags)
            for new_tag in new_tags:
                tag_to_ref[new_tag].add(ref)

    def merge(self, *args: TagsBase) -> None:
        """
        Merge a number of gmsh objects, preserving boundaries and thus allowing
        you to later select arbitrary intersections and subtractions of these
        objects.

        This method internally uses gmsh's BooleanFragments command, documented
        `here <https://gmsh.info/doc/texinfo/gmsh.html#Boolean-operations>`_.
        """
        union = None
        for x in args:
            if union is None:
                union = x
            else:
                self._merge(union, x)
                union = union | x

    def _merge(self, a: TagsBase, b: TagsBase) -> None:
        xs = list(a.get_gmsh_tags())
        ys = list(b.get_gmsh_tags())
        if not (xs and ys):
            # at least one of the operands is empty, do nothing
            return
        if len(xs) ==1 and len(ys)==1 and xs==ys:
            # Nothing to merge
            return
        dim_tags, dim_tags_mapping = self.env.fragment(
            [x.dim_tag for x in xs],
            [y.dim_tag for y in ys],
            removeObject=True,
            removeTool=True,
        )
        assert len(xs) + len(ys) == len(dim_tags_mapping)
        self._replace(
            {
                old: {GmshTag(*t) for t in new}
                for old, new in zip(itertools.chain(xs, ys), dim_tags_mapping)
            }
        )

    def remove(self, tags: TagsBase) -> None:
        """
        Delete gmsh object. Newly-invalid :class:`GmshTag` objects are removed
        to prevent accidentally referencing them, i.e.::

            tt = TagTracker()
            obj = tt.ref(...)
            tt.remove(obj)
            assert obj.get_gmsh_tag_list() == []
        """
        lst = list(tags.get_gmsh_tags())
        self.env.remove([x.dim_tag for x in lst])
        self._replace({x: set() for x in lst})


def tabulate_DG1_vertex_values(function) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two arrays, ``(coords, values)`` where::

        coords.shape == (num_cells, vertices_per_cell, geometric_dimension)
        values.shape == (num_cells, vertices_per_cell, value_dimension)
    """
    space = function.function_space()
    mesh = space.mesh()

    e = space.ufl_element()
    if e.family() != "Discontinuous Lagrange" or e.degree() != 1:
        raise AssertionError("space must be DG1")

    values = function.vector()[:]
    coords = space.tabulate_dof_coordinates()

    num_cells = mesh.num_cells()
    value_dimension = space.ufl_element().value_size()
    values = values.reshape(num_cells, -1, value_dimension)
    vertices_per_cell = values.shape[1]
    coords = coords.reshape(num_cells, vertices_per_cell, value_dimension, -1)
    coords = coords[:, :, 0, :]

    return coords, values


_UFL_CELL_TO_GMSH_POS_OBJECT_TYPE = {
    ("triangle", 2): "TT",
    ("tetrahedron", 3): "TS",
}


def _pad_to(array, shape):
    needed = False
    args = []
    for old, new in zip(array.shape, shape):
        if new is None or old == new:
            args.append((0, 0))
        elif old < new:
            needed = True
            args.append((0, new - old))
        else:
            raise AssertionError("old > new")

    if needed:
        return np.pad(array, pad_width=args, mode="constant", constant_values=0)
    else:
        return array


def export_DG1_function_to_gmsh_pos(function, fileobj) -> None:
    POS_DIM = 3

    space = function.function_space()
    mesh = space.mesh()
    ufl_cell = mesh.ufl_cell()

    # TODO: this only works for triangle/tetrahedron meshes
    # TODO: this probably only works for
    #   topological dimension == geometric dimension, I haven't really thought
    #   about it enough.

    # if ufl_cell.topological_dimension() != ufl_cell.geometric_dimension():
    #     raise AssertionError("topological != geometric dimension")

    coords, values = tabulate_DG1_vertex_values(function)

    # gmsh wants 3 dimensions
    coords = _pad_to(coords, (None, None, POS_DIM))

    if values.shape[-1] == 4:
        shape = values.shape[:-1]
        values = values.reshape(*shape, 2, 2)
        values = _pad_to(values, shape + (3, 3)).reshape(*shape, 9)
    elif values.shape[-1] != 9:
        raise AssertionError

    write = fileobj.write

    write('View "nodalMetric" {\n')

    def _s(xs):
        return ",".join(repr(x) for x in xs.reshape(-1))

    gmsh_object_type = _UFL_CELL_TO_GMSH_POS_OBJECT_TYPE[
        ufl_cell.cellname(), ufl_cell.topological_dimension()
    ]

    for cell_coord, cell_values in zip(coords, values):
        write(gmsh_object_type)
        write("(")
        write(_s(cell_coord))
        write("){")
        write(_s(cell_values))
        write("};\n")

    write("};\n")


def generate_anisotropic_mesh(
    *, xdmf_output: Path, mesh_generator: callable, density_function: callable,
    refinement_iterations: int = 0
) -> dict[str, set[int]]:
    """
    Generate an anisotropic mesh given a custom mesh generator and a custom
    anisotropic mesh density function.

    Parameters
    ----------
    xdmf_output: Path
        Path to write xdmf mesh to.
    mesh_generator: callable
        Will be called as ``mesh_generator(xdmf_path, pos_path)``
        at least twice, once with ``pos_path`` equal to None when generating an
        initial mesh, and then again with ``pos_path`` equal to the path of the
        gmsh pos file containing the anisotropic mesh density. The function
        must output the mesh density to ``xdmf_path`` and return a dictionary
        of tags and cell values.
    density_function: callable
        Will be called as
        ``density_function(mesh, cell_function, tag_values)``. Must return a
        9-dimensional FEM DG1 function representing a 3x3 anisotropic mesh
        density at every point. 
        If density_function is `None`, the initial mesh generation will be used.
    refinement_iterations: int
        Number of times to refine the mesh after the initial generation. If the 
        desired mesh density is highly nonuniform, the pos file produced on the 
        original mesh may not be refined enough to capture the desired density.
        The mesh generation process will be repeated, with a pos file generated 
        from the previous mesh. 
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        x1 = tmpdir / "m1.xdmf"
        pos2 = tmpdir / "m2.pos"
        if density_function is None:
            # we want to output this mesh
            x1 = xdmf_output
        tag_values = mesh_generator(xdmf_path=x1, pos_path=None)
        [[_, tag_values]] = tag_values.items()

        if density_function is not None:
            mesh_filename = str(x1)
            for _ in range(1 + refinement_iterations):
                with dolfin.XDMFFile(mesh_filename) as file:
                    mesh = dolfin.Mesh()
                    file.read(mesh)
                    cf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
                    file.read(cf)

                u = density_function(mesh=mesh, cell_function=cf, tag_values=tag_values)
                with pos2.open("wt", encoding="ascii") as fileobj:
                    export_DG1_function_to_gmsh_pos(u, fileobj)

                tag_values = mesh_generator(xdmf_path=Path(xdmf_output), pos_path=pos2)
                [[_, tag_values]] = tag_values.items()
                mesh_filename = str(xdmf_output)




    return tag_values


@attr.s(eq=False)
class AnisotropicPygmshGeneratorHelper:
    xdmf_path: Path = attr.ib()
    pos_path: Optional[Path] = attr.ib()

    dim: int = attr.ib(kw_only=True)
    user_data = attr.ib(default=None, kw_only=None)
    regions = attr.ib(factory=dict, kw_only=True)
    tag_tracker = attr.ib(factory=TagTracker, kw_only=True)
    characteristic_length = attr.ib(default=(0.05, 0.1), kw_only=True)
    verbose_mesh_generation: bool = attr.ib(default=False)
    gmsh_algorithm: int = attr.ib(default=7, kw_only=True)

    output_cell_values: dict[str, dict[str, frozenset]]

    @classmethod
    def prepare(cls, **kw):
        def actually_generate(xdmf_path, pos_path):
            obj = cls(xdmf_path=xdmf_path, pos_path=pos_path, **kw)
            obj._generate()
            return obj.output_cell_values

        return actually_generate

    def _generate(self):
        import pygmsh

        with pygmsh.occ.Geometry() as geom:
            self.geometry = geom

            self._define_mesh_density()
            self.user_generate()
            self._define_physical_regions()
            self._generate_and_write_mesh()

            self.geometry = None

    def _generate_and_write_mesh(self):
        dim = self.dim
        if dim not in (2, 3):
            raise AssertionError("geometric dimension must be 2 or 3")

        mesh = self.geometry.generate_mesh(
            dim=dim, algorithm=self.gmsh_algorithm, verbose=self.verbose_mesh_generation
        )
        if dim == 2:
            simudo_meshio.strip_z_component(mesh)

        simudo_meshio.keep_only_largest_dimension_cells(mesh)

        # GMSH cannot handle empty physical regions, i.e. region tags that
        # are simply empty. Here we ensure that empty regions have
        # corresponding empty sets of cell values in `output_cell_values`.
        _empty = frozenset()
        d = {k: _empty for k in self.regions}
        ret = simudo_meshio.cell_sets_to_cell_values(mesh)
        self.output_cell_values = {k: d | v for k, v in ret.items()}
        mesh.write(str(self.xdmf_path))

    def _define_mesh_density(self):
        geom = self.geometry
        if self.pos_path is None:
            a, b = self.characteristic_length
            if a is not None:
                geom.characteristic_length_min = a
            if b is not None:
                geom.characteristic_length_max = b
        else:
            gmsh.merge(str(self.pos_path))
            bg_field = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
            gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

    def _define_physical_regions(self):
        geom = self.geometry
        for k, v in self.regions.items():
            lst = v.get_pygmsh_tag_list()
            if lst:
                # workaround: add_physical needs at least one entity
                geom.add_physical(lst, k)

    def user_generate(self):
        raise NotImplementedError

class IsotropicPygmshGeneratorHelper(AnisotropicPygmshGeneratorHelper):
    '''
    A helper class to generate meshes without the BAMG algorithm.
    All of the methods from AnistropicPygmshGeneratorHelper should still work
    except we overwrite _generate_and_write_mesh to use the default mesh generation

    algorithm can be 
    1: MeshAdapt
    2: Automatic
    3: Initial mesh only
    5: Delaunay
    6: Frontal Delaunay
    7: BAMG

    '''
    algorithm = 6
    def _generate_and_write_mesh(self):
        dim = self.dim
        if dim not in (2, 3):
            raise AssertionError("geometric dimension must be 2 or 3")

        mesh = self.geometry.generate_mesh(
            dim=dim, algorithm=self.algorithm, verbose=self.verbose_mesh_generation
        )
        if dim == 2:
            simudo_meshio.strip_z_component(mesh)

        simudo_meshio.keep_only_largest_dimension_cells(mesh)

        # GMSH cannot handle empty physical regions, i.e. region tags that
        # are simply empty. Here we ensure that empty regions have
        # corresponding empty sets of cell values in `output_cell_values`.
        _empty = frozenset()
        d = {k: _empty for k in self.regions}
        ret = simudo_meshio.cell_sets_to_cell_values(mesh)
        self.output_cell_values = {k: d | v for k, v in ret.items()}
        mesh.write(str(self.xdmf_path))
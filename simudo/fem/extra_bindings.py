import dolfin

code = '''

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

#include <dolfin.h>
#include <dolfin/function/LagrangeInterpolator.h>
#include <dolfin/geometry/BoundingBoxTree.h>

namespace py = pybind11;

namespace dolfin {

void LagrangeInterpolator_interpolate_Expression
  (LagrangeInterpolator &lagrange_interpolator,
   Function &u, const Expression &u0)
{
  lagrange_interpolator.interpolate(u, u0);
}

void Function_eval_many(
  std::shared_ptr<dolfin::Function> function_,
  size_t size_x,
  size_t count,
  Eigen::Ref<Eigen::VectorXd> values,
  Eigen::Ref<const Eigen::VectorXd> x)
{
  Function &function = *function_;
  dolfin_assert(function.function_space());
  dolfin_assert(function.function_space()->mesh());
  const Mesh& mesh = *function.function_space()->mesh();
  int mesh_gdim = mesh.geometry().dim();

  /* Compute in tensor (e.g., one for scalar function) */
  const std::size_t value_size_loc = function.value_size();

  dolfin_assert(values.size() == value_size_loc * count);

  Array<double> single_x(3);

  /* clear input x vector */
  for (std::size_t k = 0; k < 3; k++)
    single_x[k] = 0;

  for (std::size_t index = 0; index < count; index++) {
    for (std::size_t k = 0; k < size_x; k++)
      single_x[k] = x[index*size_x + k];

    Array<double> single_values(value_size_loc,
      values.data() + index*value_size_loc);

    const Point point(mesh_gdim, single_x.data());

    /* Get index of first cell containing point */
    unsigned int id
      = mesh.bounding_box_tree()->compute_first_entity_collision(point);

    /* If not found, just skip */
    if (id == std::numeric_limits<unsigned int>::max())
      continue;

    /* Create cell that contains point */
    const Cell cell(mesh, id);
    ufc::cell ufc_cell;
    cell.get_cell_data(ufc_cell);

    function.eval(single_values, single_x, cell, ufc_cell);
  }
}

}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("LagrangeInterpolator_interpolate_Expression",
        &dolfin::LagrangeInterpolator_interpolate_Expression);

  m.def("Function_eval_many", [](
    py::object function,
    size_t size_x,
    size_t count,
    Eigen::Ref<Eigen::VectorXd> values,
    Eigen::Ref<const Eigen::VectorXd> x)
    {
      auto _function = function.attr("_cpp_object").cast<
        std::shared_ptr<dolfin::Function>>();
      dolfin::Function_eval_many(_function, size_x, count, values, x);
    });
}
'''

ext_module = dolfin.compile_cpp_code(code)

# ext_module = dolfin.compile_extension_module(
#     code)

__all__ = []

for name in [
        'LagrangeInterpolator_interpolate_Expression',
        'Function_eval_many']:
    globals()[name] = getattr(ext_module, name)
    __all__.append(name)

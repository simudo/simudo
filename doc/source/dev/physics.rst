
Physics
&&&&&&&

Carrier transport
=================

#. The IB is only defined within the IB material region, and not
   outside. FEniCS doesn't seem to have a way to restrict a function
   to a subset of the mesh. We therefore need to figure out how to set
   essential BC on internal dofs to exclude them from the calculation.

#. Try: Evaluate formal ``V_bc + u_bc`` to get ``w_bc``, and use that
   as the ``w`` BC.

#. Requested diagnostic: Produce plot with no majority contact fix and
   finer meshing in the y direction.

#. What if the continuity and transport equations didn't use the same
   quadrature degree for ``exp(w)``? This could lead to an
   inconsistency near a minority contact, and therefore noise.

#. Idea: Imposing ``j_u`` essential BC and iterating to impose the SRV
   BC.

#. Implement interactive 2d plots.

Optics
======

#. For Strandberg, use approximate form for radiative trapping rate
   (instead of the Fermi-Dirac degree 2 integral which they don't use
   anyway).

#. Non-square-peak absorption is important! Account for it when
   designing optical part of software.

Jacob's idea for optical fields
-------------------------------

Using a "plane wave" (all parallel) basis for the light intensity,
with one basis function per angle, means having very many light
fields, i.e. many variables (which is bad because of the ``O(N^3)``
time complexity scaling). Instead we can exploit the Lambertian
behavior of a nice back surface to use a Lambertian basis for the
light from the back contact (in addition to, for example, specular
reflection).

Presentation
============

#. Be clear when talking about CB->IB->VB processes. Recombination is
   usually reserved for CB->VB processes. The CB/VB->IB processes are
   instead referred to as "trapping" (Shockley-Read, not SRH).

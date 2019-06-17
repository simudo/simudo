
Tutorial
&&&&&&&&

pn-diode example
################

Now that you've installed Simudo, we can perform the most basic of
simulations. Namely, getting a J(V) curve for a diode (pn-junction).

Download the pn-diode_ example, then run::

  python3 pn_diode.py run

.. _pn-diode: https://github.com/simudo/simudo/raw/master/simudo/example/pn_diode/pn_diode.py

This will run the simulation, a voltage sweep from 0V to 1V. It will
produce a series of files containing spatial data (e.g.,
``output V_stepper parameter=0.00000 csvplot.csv.0``), and a CSV file
containing summary data (``output V_stepper meta.csv``).

We can produce a plot of total current as a function of applied voltage:

.. image:: /_static/tutorial-pn-diode-JV.svg

using the following code:

.. literalinclude:: ../../../simudo/example/pn_diode/pn_diode_plot.py
   :language: python


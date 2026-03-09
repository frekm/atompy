=================
Physics Submodule
=================

*atompy* provides a number of methods tailered to physics simulations / calculations.

These are further separated into submodules.



General
=======

Symbols included in the *atompy.physics* namespace can be accessed, for example,
as such:

.. code-block:: python

    import atompy as ap
    import atompy.physics as physics
    from atompy.physics import Molecule

    electron = ap.physics.Electron(...)
    atom1 = physics.Atom(...)
    atom2 = physics.Atom(...)
    molecule = Molecule([atom1, atom2])

See the :doc:`reference page <base/index>` for a documentation of everything
included in the base module.


Compton Scattering
==================

*atompy.physics.compton_scattering* groups methods related to Compton scattering.

These can be accessed like, for example:

.. code-block:: python

    import atompy as ap
    import atompy.physics.compton_scattering as compton

    tp_cross_section = ap.physics.compton_scattering.thomson_cross_section(...)
    kn_cross_section = compton.klein_nishina_cross_section(...)


See the :doc:`reference page <compton_scattering/index>` for a documentation
of everything included in this submodule.


COLTRIMS
========

*atompy.physics.coltrims* groups methods related to
Cold Target Recoil Ion Momentum Spectroscopy (COLTRIMS). 

These can be accessed like, for example:

.. code-block:: python

    import atompy as ap
    import atompy.physics.coltrims as coltrims 

    fit = coltrims.ion_tof_linear_fit(...)
    exploded = ap.physics.coltrims.coulomb_explode(...)


See the :doc:`reference page <coltrims/index>` for a documentation
of everything included in this submodule.

.. toctree::
    :hidden:

    base/index
    coltrims/index
    compton_scattering/index
=============
API reference
=============

atompy consists of a base module and a few submodules.

It is assumed that atompy was imported as

.. code-block:: python

    import atompy as ap

Base module
-----------

Everything in the base module can be accessed directly, for example,

.. code-block:: python

    vec = ap.Vector((1, 2, 3))
    hist = ap.Hist1d((1, 2, 3), (1, 2, 3, 4))
    data = ap.for_pcolormesh_from_txt("filename")

See the :doc:`reference page <base/index>` for a documentation of everything
included in the base module.

Physics module
--------------

Groups everything related to phyics and can be accessed like

.. code-block:: python

    distribution = ap.physics.mom_init_distr_elec()
    cross_section = ap.physics.compton_scattering.klein_nishina_cross_section()

See the :doc:`reference page <physics/index>` for a documentation of everything
included in this module.


.. toctree::
    :hidden:

    base/index
    physics/index



=======
Physics
=======

The :code:`pyhsics` submodule offers some theoretical modelling of various
atomic and molecular physics.

The module :ref:`section physics` is about general atomic and 
molecular physics.

The module :ref:`section compton scattering` is about Compton
scattering in particular.

Navigation
==========

.. _section physics:

:code:`atompy.physics`
----------------------

.. currentmodule:: atompy.physics

.. autosummary::
    :toctree: _autogen

    rho_p_microcanonical
    mom_init_distr_elec
    mom_init_distr_elec_mol

.. _section compton scattering:

:code:`atompy.physics.compton_scattering`
-----------------------------------------

.. currentmodule:: atompy.physics.compton_scattering

.. autosummary::
    :toctree: _autogen

    thomson_cross_section
    klein_nishina_cross_section
    compton_photon_energy_out
    scattering_angle_distr
    mom_final_distr_photon
    mom_final_distr_elec
    mom_transfer_approx
    stretch_Compton_electron_onto_sphere
    subtract_binding_energy
    calculate_Q_neglecting_mom_init


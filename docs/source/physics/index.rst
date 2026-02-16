=================
Physics Submodule
=================

*atompy* provides a number of methods tailered to physics simulations / calculations.

Access these like

.. code-block:: python

    import atompy as ap

    ap.physics                    # general methods
    ap.physics.compton_scattering # methods related to Compton scattering
    ap.physics.coltrims           # methods related to COLTRIMS


General
=======

.. currentmodule:: atompy.physics

.. autosummary::
    :toctree: _autogen

    subtract_binding_energy
    rho_p_microcanonical
    mom_init_distr_elec
    mom_init_distr_elec_mol


Compton Scattering
==================

.. currentmodule:: atompy.physics.compton_scattering

.. autosummary::
    :toctree: _autogen

    thomson_cross_section
    klein_nishina_cross_section
    scattering_angle_distr
    mom_final_distr_elec
    mom_final_distr_photon
    mom_final_distr_photon_var
    mom_transfer_approx
    stretch_Compton_electron_onto_sphere
    compton_photon_energy_out
    calculate_Q_neglecting_mom_init


COLTRIMS
========

.. currentmodule:: atompy.physics.coltrims

.. autosummary::
    :toctree: _autogen

    ion_tof_linear_fit
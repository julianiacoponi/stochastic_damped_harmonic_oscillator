PARAMETER(&
    ! basic physical constants
    ! use at least 6 digits, from https://physics.nist.gov/cuu/Constants/
    pi=dacos(-1.d0), &
    hbar=1.054571817d-34, & ! Planck's reduced constant (Joules/second)
    speed_of_light=299792498, & ! metres/second
    kB=1.380649d-23, & ! Boltzmann's constant (Joules/Kelvin)
    vacuum_permittivity=8.854187d-12, & ! permittivity of free space (Farad/metre)

    ! conversion constants
    Hz=pi + pi, &  ! Hz to radians/s
    kHz=1.d3 * Hz, &  ! kHz to radians/s
    degrees=pi / 180.d0, &  ! degrees to radians
    nanometres=1.d-9, &  ! metres to nanometres
    microseconds=1.d-6, &  ! seconds to microseconds
    mbar=100.d0, &  ! millibar to Pascals

    ! propertires of air
    air_diameter=0.372 * nanometres, &  ! diameter of air molecule
    air_viscosity=18.27d-6, &  ! viscosity of air, kg/m/s = Pa.s
    air_mass=4.8d-26, &  ! mass of air molecule
    air_temperature=300.d0, & ! Kelvin
    air_pressure=6.5d-3 * mbar, &

    ! parameters of Jon and Antonio's setup
    bead_radius=70 * nanometres, &
    bead_density=1850.d0, & ! kilograms/metre^3
    bead_permittivity=1.98d0, & ! dimensionless

    omega_x = 120.d0 * kHz, &
    omega_y = 140.d0 * kHz, &

    x0=0.d0, &  ! initial position
    v0=0.d0)  ! initial velocity
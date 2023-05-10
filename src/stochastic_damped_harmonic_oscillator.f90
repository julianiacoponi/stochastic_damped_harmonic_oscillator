MODULE stochastic_damped_harmonic_oscillator

    ! https://github.com/AshleySetter/optoanalysis/blob/master/SDE_Solution_Derivation.ipynb

    ! Equation of motion: a + Γv + ω**2*x = F/m
    ! a = dv/dt
    ! v = dx/dt
    ! F = stochastic force = sqrt(2Γ * kbT / m) * dW/dt = wiener_kick * dW/dt
    ! dW represents the differential bump of a random Wiener process


    ! dX(t) = a(t,X)dt + bdW
    ! X is a column vector = (x, v)^T
    ! a(t, X) = (v, -Γv -ω**2*x)^T
    ! b = (0, wiener_kick)^T

    ! This is a vector Ito process which solves the two stochastic differential equaitons (SDEs)
    ! dx = vdt
    ! dv = -Γv -ω**2*x + wiener_kick dW

    USE stdlib_random, ONLY: dist_rand, random_seed
    USE stdlib_stats_distribution_normal, ONLY: norm => rvs_normal

    IMPLICIT NONE

    ! parameter file with input values
    double precision::pi, hbar, speed_of_light, kB, vacuum_permittivity
    double precision::Hz, kHz, degrees, nanometres, microseconds, mbar
    double precision::air_diameter, air_viscosity, air_mass, air_temperature, air_pressure
    double precision::bead_radius, bead_density, bead_permittivity
    double precision::omega_x, omega_y
    double precision::x0, v0
    INCLUDE 'C15_PARAMS.h'

    double precision::omega
    double precision::bead_mass
    double precision::numerical_prefactor, physical_prefactor, gas_damping
    double precision::feedback_damping, damping

    integer::num_samples
    PARAMETER(num_samples=1e7)
    double precision, DIMENSION(num_samples, 2)::X_array
    double precision, DIMENSION(num_samples)::noise_array

    PRIVATE

    PUBLIC :: &
        Physics, &
        TimeTrace, &
        fill_time_array, &
        new_noise_array

CONTAINS


    SUBROUTINE Physics

        PRINT *, 'omega_x', omega_x / kHz, 'kHz'
        PRINT *, 'omega_y', omega_y / kHz, 'kHz'

        PRINT *, 'air_diameter', air_diameter, 'metres'
        PRINT *, 'air_viscosity', air_viscosity, 'kg/m/s'
        PRINT *, 'air_mass', air_mass, 'kg'
        PRINT *, 'air_temperature', air_temperature, 'Kelvin'
        PRINT *, 'air_pressure', air_pressure / mbar, 'mbar'

        bead_mass = bead_density * (4.d0/3)*pi * bead_radius**3
        PRINT *, 'bead_mass ', bead_mass, 'kg'

        numerical_prefactor = (0.619d0 * 9*pi / sqrt(2.d0))
        physical_prefactor = (air_viscosity * air_diameter**2 / (bead_density * kB))
        gas_damping = numerical_prefactor * physical_prefactor * (air_pressure / (air_temperature * bead_radius))
        feedback_damping = 3.d0 * kHz
        damping = gas_damping + feedback_damping

        PRINT *, 'gas_damping ', gas_damping / kHz, 'kHz'
        PRINT *, 'feedback_damping ', feedback_damping / kHz, 'kHz'
        PRINT *, 'damping ', damping / kHz, 'kHz'

        CALL new_noise_array()
        CALL system('mkdir output')

    END SUBROUTINE Physics


    SUBROUTINE TimeTrace(axis, noise_angle)
        character(len=1)::axis
        integer::noise_angle

        integer::n, S, rand_num, trace
        double precision::time_step, delta
        double precision::dW, angle_factor
        double precision::t, x, v
        double precision::K1x, K1v, K2x, K2v
        character(len=100)::noise_angle_str  ! always give more length, and then use trim()

        WRITE(noise_angle_str, '(I0)') noise_angle  ! converts integer to string

        CALL system('mkdir output/angle_'//trim(noise_angle_str))

        IF (axis == 'x') THEN
            omega = omega_x
            angle_factor = cos(noise_angle * degrees)
            trace = noise_angle
        ELSE IF (axis == 'y') THEN
            omega = omega_y
            angle_factor = sin(noise_angle * degrees)
            trace = noise_angle * 20
        END IF

        OPEN(trace, file='output/angle_'//trim(noise_angle_str)//'/'//trim(axis)//'trace.dat', status='unknown')

        time_step = 0.1 * microseconds
        X_array(1, 1) = x0
        X_array(1, 2) = v0

        delta = sqrt(time_step)

        DO n=1, num_samples

            ! IF (mod(n, 10000) == 0) THEN
            !     PRINT *, 'Loop #', n
            ! END IF

            t = time_step * (n - 1)
            dW = angle_factor * delta * noise_array(n)

            ! S = np.random.choice([+1, -1])
            S = 1.d0
            rand_num = dist_rand(1)
            IF (rand_num < 0) THEN
                S = -1.d0
            END IF

            x = X_array(n, 1)
            v = X_array(n, 2)
            ! don't worry about recording the velocity for now
            WRITE(trace, *) x

            ! ensures the final loop writes the above values without trying to calculate beyond the sample limit
            IF (n /= num_samples) THEN
                K1x = a_x(t, x, v)*time_step + b_x(t, x, v)*(dW - S*delta)
                K1v = a_v(t, x, v)*time_step + b_v(t, x, v)*(dW - S*delta)

                x = x + K1x
                v = v + K1v

                K2x = a_x(t, x, v)*time_step + b_x(t, x, v)*(dW + S*delta)
                K2v = a_v(t, x, v)*time_step + b_v(t, x, v)*(dW + S*delta)

                X_array(n + 1, 1) = X_array(n, 1) + 0.5d0 * (K1x + K2x)
                X_array(n + 1, 2) = X_array(n, 2) + 0.5d0 * (K1v + K2v)
            END IF

        END DO

        CLOSE(trace)

    END SUBROUTINE TimeTrace

    ! separated to avoid duplicating this in every output file
    SUBROUTINE fill_time_array
        integer::n, time_array_file
        double precision::time_step

        time_array_file = 181
        OPEN(time_array_file, file='output/time_array.dat', status='unknown')

        time_step = 0.1 * microseconds
        DO n=1, num_samples
            WRITE(time_array_file, *) time_step * (n - 1)
        END DO

        CLOSE(time_array_file)

    END SUBROUTINE fill_time_array


    SUBROUTINE new_noise_array
        noise_array = norm(0.0, 1.0, num_samples)
    END SUBROUTINE

    ! deterministic part of dx = v.dt
    double precision FUNCTION a_x(t, x, v)
        double precision, INTENT (IN)::t, x, v
        a_x = v
    END FUNCTION

    ! stochastic part of dx = v.dt
    double precision FUNCTION b_x(t, x, v)
        double precision, INTENT (IN)::t, x, v
        b_x = 0.d0
    END FUNCTION

    ! deterministic part of dv = a.dt + b.dW
    double precision FUNCTION a_v(t, x, v)
        double precision, INTENT (IN)::t, x, v
        a_v = -damping * v - omega**2 * x
    END FUNCTION

    ! stochastic part of dv = a.dt + b.dW
    double precision FUNCTION b_v(t, x, v)
        double precision, INTENT (IN)::t, x, v
        b_v = sqrt(2*damping * kB*air_temperature / bead_mass)
    END FUNCTION

END MODULE stochastic_damped_harmonic_oscillator

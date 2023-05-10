PROGRAM main
    USE stochastic_damped_harmonic_oscillator, ONLY: &
        Physics, &
        TimeTrace, &
        fill_time_array, &
        new_noise_array

    IMPLICIT NONE

    integer::angle_index, num_angles, noise_angle

    CALL Physics()
    ! noise_angle = 45
    ! CALL TimeTrace('x', noise_angle)
    ! CALL TimeTrace('y', noise_angle)

    ! only 1 time array needed :)
    CALL fill_time_array()

    num_angles = 181
    DO angle_index=1, num_angles
        noise_angle = (angle_index - 1) * 1
        CALL TimeTrace('x', noise_angle)
        CALL TimeTrace('y', noise_angle)

        ! this ensures each pair of traces gets a fresh set of noise
        CALL new_noise_array()
    END DO

END PROGRAM main

#!/opt/homebrew/bin/python3

import math
import pickle

import scipy.constants
import scipy.signal
import scipy.integrate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# basic physical constants
π = np.pi
kb = scipy.constants.Boltzmann  # Joules/Kelvin
e = scipy.constants.elementary_charge # Coloumbs

# conversion constants
Hz = π + π  # Hz to radians/s
kHz = 1e3 * Hz  # kHz to radians/s
degrees = π  / 180  # degrees to radians
nm = 1e-9  # metres to nanometres
ms = 1e-3  # seconds to milliseconds
μs = 1e-6  # seconds to microseconds
mbar = 100 # mbar to Pascals

# physical constants of air as a gas
d_gas = 0.372*nm  # diameter of air molecule
ξ = 18.27e-6  # viscosity of air, kg/m/s = Pa.s
m_gas = 4.8e-26  # mass of air molecule

# constants to be used across scripts
NUM_SAMPLES = 1e6
TIME_STEP = 0.1*μs

# make NUM_ANGLES a multiple of psutil.cpu_count() ! In macoponi's case, this is 8.
ANGLE_START, ANGLE_STEP, NUM_ANGLES = 0, 1, 181
ANGLE_END = ANGLE_START + NUM_ANGLES * ANGLE_STEP
ANGLES = list(range(ANGLE_START, ANGLE_END, ANGLE_STEP))

# defaults for use in large simulations for correlated noise x,y trace pairs
LOW_PRESSURE = 1e-3*mbar
FB_DAMPING = 3*kHz  # Antonio suggests this will give the 'inverted bow' shape :-)
OMEGA_X, OMEGA_Y = 120*kHz, 140*kHz

class Physics:
    """
    https://github.com/AshleySetter/optoanalysis/blob/master/SDE_Solution_Derivation.ipynb

    Equation of motion: a + Γv + ω**2*x = F/m
    a = dv/dt
    v = dx/dt
    F = stochastic force = sqrt(2Γ * kbT / m) * dW/dt = wiener_kick * dW/dt
    dW represents the differential bump of a random Wiener process


    dX(t) = a(t,X)dt + bdW
    X is a column vector = (x, v)^T
    a(t, X) = (v, -Γv -ω**2*x)^T
    b = (0, wiener_kick)^T

    This is a vector Ito process which solves the two stochastic differential equaitons (SDEs)
    dx = vdt
    dv = -Γv -ω**2*x + wiener_kick dW
    """

    def __init__(
            self,

            # constants of the (assumed spherical) silica particle
            r=70*nm,
            ρ=1850.0,

            ω=120*kHz,

            P=10*mbar,
            T=300.0,

            # TODO: add in parameters to get this, i.e. q, E, and a transfer function H (see notes)
            # q=10*e,
            # E=?
            feedback_damping=0.0,

            x0=0.0,
            v0=0.0,
        ):


        self.r = r  # particle radius (metres)
        self.ρ = ρ  # particle density (kg/m**3)
        self.ω = ω  # particle oscillation frequency (radians/s)
        self.P = P  # gas pressure (Pascals)
        self.T = T  # gas temperature (Kelvin)

        # initial conditions for the position and velocity of the particle
        self.x0 = x0  # metres
        self.v0 = v0  # metres/s

        self.m = ρ * (4/3)*π * r**3  # particle mass (kg)

        # Millen, Monteiro IOP Levitated Optomech Review
        # https://doi.org/10.1088/1361-6633/ab6100 page 3
        # TODO: not sure why broken?
        # self.gas_damping = (8/3) * np.sqrt(( 2*self.m / (π*kb*T) )) * r**2 * P

        # from Ulbricht 2017 Optica paper
        # https://doi.org/10.1364/JOSAB.34.001421
        numerical_prefactor = 0.619 * 9*π / np.sqrt(2)
        physical_prefactor = (ξ * d_gas**2) / (ρ * kb)
        self.gas_damping = numerical_prefactor * physical_prefactor * ( P / (T * r) )  # gas damping (radians/s)

        self.feedback_damping = feedback_damping
        self.Γ = self.gas_damping + self.feedback_damping

        # Equipartition theorem
        # ½mω²〈x²〉 = ½kb*T
        # Area under PSD = 〈x²〉
        # Therefore expecting T = (mω² / kb) * Area
        self.equipartition_factor = self.m*ω**2 / kb

        return

    def a_x(self, t, x, v):
        return v

    def b_x(self, t, x, v):
        return 0

    def a_v(self, t, x, v):
        return -self.Γ * v - self.ω**2 * x

    def b_v(self, t, x, v):
        # the Wiener kick !
        # This is the stochastic force / mass
        return np.sqrt(2*self.gas_damping * kb*self.T / self.m)

    def deterministic(self, t, X):
        x, v = X
        return np.array([self.a_x(t, x, v), self.a_v(t, x, v)])

    def stochastic(self, t, X):
        x, v = X
        return np.array([self.b_x(t, x, v), self.b_v(t, x, v)])


# NOTE: since the Noise needs the properties of the time trace to be defined, it can't be part of Physics()
# Also, since we want to give the same *random* noise to (at least) two separate traces, it needs to be outside of TimeTrace()
class Noise:
    """ 1D noise, acting in a 2D plane defined by the angle """
    def __init__(self, num_samples, Δt, angle=0.0*degrees):
        self.angle = angle
        𝛿 = np.sqrt(Δt)
        self.dW_array = 𝛿 * np.random.normal(0, 1, int(num_samples))
        return


class TimeTrace:
    """ 1D time trace """
    properties = [
        'Axis',
        'Time Length (ms)',
        'Num Samples',
        'Time Step (μs)',
        'Sample Rate (MHz)',

        # physics
        'Trap Frequency (kHz)',
        'Pressure (mbar)',
        'Temperature (K)',
        'Temperature via 〈x²〉(K)',
        'Gas Damping (kHz)',
        'Feedback Damping (kHz)',
        'Noise Angle (deg)',

        '<TimeTrace instance>',
        'id(<TimeTrace instance>)',

        '<Noise instance>',
        'id(<Noise instance>)',
    ]

    def __init__(
        self,
        physics,
        num_samples,
        Δt,
        **kwargs,
    ):
        # default
        self.axis = kwargs.get('axis', 'x')

        self.physics = physics
        self.num_samples = num_samples
        self.Δt = Δt

        self.time_array, self.displacement, self.velocity = self.vector_runge_kutta(
            physics,
            **kwargs,
        )
        self.PSD_list = []

    @property
    def time_length(self):
        return self.num_samples * self.Δt

    @property
    def sample_rate(self):
        return 1 / self.Δt

    @property
    def displacement_variance(self):
        return np.var(self.displacement)

    @property
    def temperature_via_variance(self):
        # Equipartition theorem
        # ½mω²〈x²〉 = ½kb*T
        # Area under PSD = 〈x²〉
        # Therefore expecting T = (mω² / kb) * Area
        return self.physics.equipartition_factor * self.displacement_variance

    @property
    def info(self):
        trace_repr = (
            f'\nTimeTrace'
            f'(\n   axis={self.axis}'
            f',\n   time_length={self.time_length / ms:.3}ms'
            f',\n   num_samples={self.num_samples:.3}'
            f',\n   Δt={self.Δt / μs:.3}μs'
            f',\n   sample_rate={self.sample_rate / 1e6:.3}MHz'

            # physics
            f',\n   ω={self.physics.ω / kHz:.3}kHz'
            f',\n   P={self.physics.P / mbar:.3}mbar'
            f',\n   T={self.physics.T:.3}K'
            f',\n   T via〈x²〉={self.physics.T:.3}K'
            f',\n   gas_damping={self.physics.gas_damping / kHz:.3}kHz'
            f',\n   feedback_damping={self.physics.feedback_damping / kHz:.3}kHz'
            f',\n   noise.angle={self.noise.angle / degrees:.3}deg'
            f',\n   id(self)={id(self)}'
            f'\n)'
        )
        return trace_repr

    def vector_runge_kutta(
        self,
        physics,
        **kwargs,
    ):
        """ Generates the time trace using the vector Runge-Kutta numerical method """
        a = physics.deterministic
        b = physics.stochastic

        num_samples = int(self.num_samples)
        Δt = self.Δt

        X = np.zeros([num_samples, 2])
        # initial conditions for the position and velocity
        X[0, 0] = physics.x0
        X[0, 1] = physics.v0

        noise = kwargs.get('noise', None)
        if noise is None:
            self.noise = Noise(num_samples, Δt)
        else:
            self.noise = noise

        angle = self.noise.angle
        angle_factor = np.cos(angle)
        if self.axis == 'y':
            angle_factor = np.sin(angle)

        dW_array = angle_factor * self.noise.dW_array
        𝛿 = np.sqrt(Δt)
        time_array = np.arange(num_samples * Δt, step=Δt)
        for n, t in enumerate(time_array[:-1]):
            S = np.random.choice([+1, -1])
            dW = dW_array[n]

            K1 = a(t, X[n])*Δt + b(t, X[n])*(dW - S*𝛿)
            K2 = a(t, X[n] + K1)*Δt + b(t, X[n] + K1)*(dW + S*𝛿)

            X[n + 1] = X[n] + 0.5 * (K1 + K2)

        x = X[:, 0]
        v = X[:, 1]

        return time_array, x, v

    def plot_displacement(self):
        plt.plot(self.time_array / ms, self.displacement / nm)
        plt.xlabel('t (milliseconds)')
        plt.ylabel(f'{self.axis}-position (nm)')
        plt.title(f'Total trace time = {self.time_length:.3}s')
        plt.show()

    def plot_momentum(self):
        plt.plot(self.time_array / ms, self.physics.m*self.velocity)
        plt.xlabel('t (milliseconds)')
        plt.ylabel('momentum (kgm/s)')
        plt.show()

    def estimate_PSD(self, **kwargs):
        self.PSD_list.append(SpectralDensityEstimate(
            self,
            **kwargs,
        ))

    @property
    def PSDs(self):
        self.PSDs_database = pd.DataFrame(
            [[
                psd.temperature,
                psd.num_segments,
                psd.segment_width,
                psd.Δf,
                psd.segment_time,
                len(psd.freq_range),
                len(psd.positive_freqs),
            ] for psd in self.PSD_list],
            columns=SpectralDensityEstimate.properties
        )
        return self.PSDs_database


class SpectralDensityEstimate:
    """
    Estimates the:
    - PSD (power spectral density), if one time trace is provided
    - CSD (cross-correlation spectral density), if two time traces are provided
    """

    properties = [
        'Temperature via Area (K)',
        'Num Segments',
        'Segment Width',
        'Frequency Step (Hz)',
        'Segment Time (s)',
        'Length of Freq. Range',
        'Length of Positive Freq. Range',
    ]

    def __init__(self, *time_traces, **kwargs):
        self.debug = kwargs.get('debug', False)

        if len(time_traces) == 1:
            self.type = 'PSD'
            self.trace = time_traces[0]
        elif len(time_traces) == 2:
            self.type = 'CSD'
            self.trace_A, self.trace_B = time_traces
            # NOTE: assumes A and B have the same #samples and time step
            self.trace = self.trace_A

        self.num_segments = kwargs.pop('num_segments', 50)
        num_samples = self.trace.num_samples
        sample_rate = self.trace.sample_rate
        self.segment_width = math.floor(num_samples / self.num_segments)

        # frequency step size for each segmented PSD estimate, equivalent to (num_segments / trace.time_length)
        # also: the index-to-frequency conversion factor, in Hz
        self.Δf = sample_rate / self.segment_width

        # time length of each segment, equivalent to (segment_width * Δt)
        # also: the frequency-to-index conversion factor, in seconds
        self.segment_time = 1 / self.Δf

        # from John / Antonio's coded
        # self.window = scipy.signal.windows.blackmanharris(int(self.segment_width)) / ( 287/800 ) / np.sqrt(2)
        self.window = 1

        # method to integrate the PSD with
        # other choice might be: scipy.integrate.simpson
        self.area_method = np.trapz

        # leave this at the end as it creates the CSD
        self.values = self.estimate_spectral_density(**kwargs)
        self.positive_values = self.positive_range(self.values)
        self.positive_freqs = self.positive_range(self.freq_range)

    def positive_range(self, values):
        """ returns the positive frequency range of e.g. the PSD, or its frequency bins """
        return values[:math.floor(self.segment_width /  2)]

    def estimate_spectral_density(
            self,

            # possible values for the norm kwarg of np.fft.fft():
            # 'forward' gives the factor of 1/num_samples to the FFT only
            # so a factor of 1/num_samples**2 to the PSD estimation

            # 'backward' gives the factor of 1/num_samples to the inverse FFT only
            # so no factor added to the PSD estimation

            # 'ortho' *symmetrises* the FFT and its inverse, applying 1/sqrt(num_samples) to each
            # so a factor of 1/num_samples to the PSD estimation
            fft_norm='ortho',

            # this scaling boils down to a factor of 1 / (segment_width * num_segments) = 1 / num_samples
            Bartlett_scaling=True,

            # are we concerned with the area of just the postive frequency part of the PSD?
            # If so, set to true!
            integration_limits_0_to_inf=False,

            # returns the imaginary part of the PSD
            imaginary_CSD=False,
        ):
        """
        Returns the spectral density estimate as an array
        """
        self.fft_norm = fft_norm
        self.Bartlett_scaling = Bartlett_scaling
        self.integration_limits_0_to_inf = integration_limits_0_to_inf
        self.imaginary_CSD = imaginary_CSD

        Δt = self.trace.Δt
        num_samples = self.trace.num_samples
        num_segments = self.num_segments
        segment_width = self.segment_width

        self.segment_width = math.floor(num_samples / num_segments)
        self.freq_range = np.fft.fftfreq(segment_width, d=Δt)

        # initialise the arrays to sum each data segment's periodogram
        segment_SDs = np.zeros(self.segment_width)
        start_index = 0
        for segment_index in range(num_segments):
            # Bartlett wiki:
            # "2. For each segment, compute the periodogram by computing the discrete Fourier
            # transform (DFT version which does not divide by M), then computing the squared magnitude
            # of the result and dividing this by M."
            end_index = (segment_index + 1) * self.segment_width

            if self.type == 'PSD':
                segment = self.trace.displacement[start_index:end_index] * self.window
                segment_fft = np.fft.fft(segment, norm=fft_norm)
                segment_SD = abs(segment_fft)**2

            elif self.type == 'CSD':
                segment_A = self.trace_A.displacement[start_index:end_index] * self.window
                segment_B = self.trace_B.displacement[start_index:end_index] * self.window
                segment_fft_A = np.fft.fft(segment_A, norm=fft_norm)
                segment_fft_B = np.fft.fft(segment_B, norm=fft_norm)

                real_or_imag = np.imag if self.imaginary_CSD else np.real
                segment_SD = real_or_imag(segment_fft_A * np.conj(segment_fft_B))

            if Bartlett_scaling:
                segment_SDs += segment_SD / segment_width
            else:
                segment_SDs += segment_SD

            start_index = end_index

        self.integration_fudge = 2 if integration_limits_0_to_inf else 1

        # Bartlett wiki:
        # "3. Average the result of the periodograms above for the K data segments. The
        # averaging reduces the variance, compared to the original N point data segment. The end
        # result is an array of power measurements vs. frequency "bin"."

        if Bartlett_scaling:
            return self.integration_fudge * self.normalisation * segment_SDs / num_segments

        return self.integration_fudge * self.normalisation * segment_SDs

    @property
    def normalisation(self):
        """ The normalisation factor needed for the estimated PSD area to represent temperature """
        num_samples = self.trace.num_samples
        Δt = self.trace.Δt
        num_segments = self.num_segments

        # TODO: Why these fudges?
        # for clarity, in terms of the original 3 inputs:
        # Δf = num_segments / (Δt * num_samples)
        #    = (1/Δt) / (num_samples / num_segments)
        #    = sample_rate / segment_width
        # 1/Δf = (Δt * num_samples) / num_segments
        #      = segment_width * Δt
        #      = segment_time
        Bartlett_scaling = self.Bartlett_scaling
        fft_norm = self.fft_norm

        # NOTE: each step from backward -> ortho -> forward effectively adds a factor 1/num_samples
        # to abs(segment_fft)**2, because it adds 1/sqrt(num_samples) to the fft.
        # HOWEVER: the fudge factors imply a factor of 1/num_segments is also applied.
        # Unsure why!
        # This wouldn't be noticed if num_segments=1, i.e. a direct periodogram as a PSD estimate.

        if Bartlett_scaling:
            if fft_norm == 'backward':
                # since Bartlett's scaling is effectively 1 / num_samples,
                # this effectively mimics the factor for norm='backward' with no Bartlett scaling
                self.fudge = 'Δt'
                return Δt
            if fft_norm == 'ortho':
                # Why divide Bartlett's method by a factor of the frequency bin size?
                # 1/Δf = (Δt * num_samples) / num_segments
                self.fudge = '(Δt*num_samples)/num_segments'
                return (Δt * num_samples) / num_segments
            if fft_norm == 'forward':
                # since Bartlett's scaling is effectively 1 / num_samples,
                # and 'forward' adds another factor of 1 / num_samples
                # this effectively mimics the factor for norm='forward' with no Bartlett scaling
                self.fudge = 'Δt*num_samples**2/num_segments**2'
                return Δt * num_samples**2 / num_segments**2

        # no need for else after return :-)
        if fft_norm == 'backward':
            # this is effectively the overall factor you apply to the abs(segment_fft)**2
            # to get the right temperature...
            self.fudge = 'Δt/num_samples'
            return Δt / num_samples
        elif fft_norm == 'ortho':
            self.fudge = 'Δt/num_segments'
            return Δt / num_segments
        elif fft_norm == 'forward':
            self.fudge = 'Δt*num_samples/num_segments**2'
            return Δt * num_samples / num_segments**2

    @property
    def area(self):
        # TODO: figure out integration limit choices...
        if self.integration_limits_0_to_inf:
            return self.area_method(self.positive_values, self.positive_freqs)
        return self.area_method(self.values, self.freq_range)

    @property
    def temperature(self):
        # Equipartition theorem
        # ½mω²〈x²〉 = ½kb*T
        # Area under PSD = 〈x²〉
        # Therefore expecting T = (mω² / kb) * Area
        return self.trace.physics.equipartition_factor * self.area

    def get_kHz_plot_points(self, lower_kHz, upper_kHz):
        # NOTE: Δf converts index-to-frequency, so 1/Δf converts frequency-to-index
        plot_lower = math.floor(lower_kHz * 1e3 / self.Δf)
        plot_upper = math.floor(upper_kHz * 1e3 / self.Δf)

        freq_plot_values = self.positive_freqs[plot_lower:plot_upper] * 1e-3
        spectral_density_plot_values = self.positive_values[plot_lower:plot_upper]

        return freq_plot_values, spectral_density_plot_values

    def plot_vs_kHz(self, lower_kHz, upper_kHz, semilogy=False):
        x_points, y_points = self.get_kHz_plot_points(lower_kHz, upper_kHz)

        num_samples = self.trace.num_samples
        Δt = self.trace.Δt
        num_segments = self.num_segments

        if self.debug:
            trace_temperature = self.trace.physics.T
            psd_temperature = self.temperature
            print(
                f'\nfft_norm={self.fft_norm}'
                f'\nBartlett_scaling={self.Bartlett_scaling}'
                f'\nfudge={self.fudge}'
                f'\nfudge_factor={self.fudge_factor}'
                f'\nlog10(T_gas/T)={math.log10(trace_temperature/psd_temperature):.3}'
                f'\nlog10(num_samples)={math.log10(num_samples):.3}'
                f'\nlog10(Δt)={math.log10(Δt):.3}'
                f'\nlog10(num_segments)={math.log10(num_segments):.3}'
                f'\nlog10(Δf)={math.log10(self.Δf):.3}'
            )

        if semilogy:
            plt.semilogy(x_points, y_points)
        else:
            plt.plot(x_points, y_points)

        plt.grid('both')
        plt.title(
            f'(#samp, Δt, #seg)={num_samples, Δt, num_segments}'
            f'; angle={self.trace.noise.angle / degrees:.3}deg'
        )
        plt.xlabel('Frequency (kHz)')
        plt.ylabel(f'{self.type} (m²/Hz)')
        plt.fill_between(x_points, y_points, 0, color='blue', alpha=0.1)
        plt.tight_layout()
        plt.show()


def create_trace_df(trace_list):
    return pd.DataFrame(
        [[
            trace.axis,
            round(trace.time_length / ms, 3),
            round(trace.num_samples, 3),
            round(trace.Δt / μs, 3),
            round(trace.sample_rate / 1e6, 3),

            # physics
            round(trace.physics.ω / kHz,  3),
            round(trace.physics.P / mbar, 3),
            round(trace.physics.T, 3),
            round(trace.temperature_via_variance, 3),
            round(trace.physics.gas_damping / kHz, 3),
            round(trace.physics.feedback_damping / kHz, 3),
            round(trace.noise.angle / degrees, 3),

            trace,
            id(trace),
            trace.noise,
            id(trace.noise),
        ] for trace in trace_list],
        columns=TimeTrace.properties
    )


def save_variable_to_pickle(variable, save_name='default.pickle'):
    save_name = 'pickle/' + save_name
    # 'wb' = write binary
    file = open(save_name, 'wb')
    pickle.dump(variable, file)
    file.close()


def load_variable_from_pickle(save_name):
    save_name = 'pickle/' + save_name
    # 'rb' = read binary
    file = open(save_name, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def generate_xy_traces_with_angle(num_samples, Δt, angle):
    correlated_noise_with_angle = Noise(num_samples, Δt, angle=angle)
    return [
        TimeTrace(
            Physics(P=LOW_PRESSURE, feedback_damping=FB_DAMPING, ω=OMEGA_X),
            num_samples, Δt,
            noise=correlated_noise_with_angle,
        ),
        TimeTrace(
            Physics(P=LOW_PRESSURE, feedback_damping=FB_DAMPING, ω=OMEGA_Y),
            num_samples, Δt,
            noise=correlated_noise_with_angle,
            axis='y',
        ),
    ]

# adapted from:
# https://stackoverflow.com/questions/44987972/heatmap-with-matplotlib
def plot_heatmap(x, y, intensity, title='CSD (m²/Hz)'):
    x, y = np.meshgrid(x, y)
    plt.pcolormesh(x, y, intensity)
    plt.title(title)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Angle (degrees)')
    plt.colorbar()
    plt.show()
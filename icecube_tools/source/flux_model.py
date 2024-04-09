import numpy as np
import mpmath as mp
from abc import ABC, abstractmethod

from .power_law import BoundedPowerLaw
from .power_law import BrokenBoundedPowerLaw
from .power_law import BoundedPowerLawExpCutoff
from .power_law import BoundedPowerLawSubexpCutoff

"""
Module for simple flux models used in 
neutrino detection calculations
"""


class FluxModel(ABC):
    """
    Abstract base class for flux models.
    """

    @abstractmethod
    def spectrum(self):

        pass

    @abstractmethod
    def integrated_spectrum(self):

        pass

    @abstractmethod
    def redshift_factor(self, z):

        pass


class PowerLawFlux(FluxModel):
    """
    Power law flux models.
    """

    def __init__(
        self,
        normalisation,
        normalisation_energy,
        index,
        lower_energy=1e2,
        upper_energy=np.inf,
    ):
        """
        Power law flux models.

        :param normalisation: Flux normalisation [GeV^-1 cm^-2 s^-1 sr^-1] or [GeV^-1 cm^-2 s^-1] for point sources.
        :param normalisation energy: Energy at which flux is normalised [GeV].
        :param index: Spectral index of the power law.
        :param lower_energy: Lower energy bound [GeV].
        :param upper_energy: Upper enegry bound [GeV], unbounded by default.
        """

        super().__init__()

        self._normalisation = normalisation

        self._normalisation_energy = normalisation_energy

        self._index = index

        self._lower_energy = lower_energy
        self._upper_energy = upper_energy

        self.power_law = BoundedPowerLaw(
            self._index, self._lower_energy, self._upper_energy
        )


    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO depending on flux_type.
        """

        if isinstance(energy, np.ndarray):
            nans = np.nonzero(((energy < self._lower_energy) | (energy > self._upper_energy)))
            output = self._normalisation * np.power(
            energy / self._normalisation_energy, -self._index
            )
            output[nans] = 0.
            return output

        else:
            if (energy < self._lower_energy) or (energy > self._upper_energy):

                return 0.
            else:
                return self._normalisation * np.power(
                    energy / self._normalisation_energy, -self._index
                )


    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        \int spectrum dE over finite energy bounds.

        :param lower_energy_bound: [GeV]
        :param upper_energy_bound: [GeV]
        """

        """
        if lower_energy_bound < self._lower_energy and upper_energy_bound < self._lower_energy:
            return 0
        elif lower_energy_bound < self._lower_energy and upper_energy_bound > self._lower_energy:
            lower_energy_bound = self._lower_energy

        if upper_energy_bound > self._upper_energy and lower_energy_bound > self._upper_energy:
            return 0
        elif upper_energy_bound > self._upper_energy and lower_energy_bound < self._upper_energy:
            upper_energy_bound = self._upper_energy
        """
        #works with np.ndarrays!

        lower_energy_bound = np.atleast_1d(lower_energy_bound).copy()
        upper_energy_bound = np.atleast_1d(upper_energy_bound).copy()
        # check for bounds being sensible
        assert np.all(upper_energy_bound - lower_energy_bound >= 0.)

        lower_energy_bound[lower_energy_bound < self._lower_energy] = self._lower_energy
        upper_energy_bound[upper_energy_bound > self._upper_energy] = self._upper_energy
        
        norm = self._normalisation / (
            np.power(self._normalisation_energy, -self._index) * (1 - self._index)
        )

        output = norm * (
            np.power(upper_energy_bound, 1 - self._index)
            - np.power(lower_energy_bound, 1 - self._index)
        )

        output[upper_energy_bound <= self._lower_energy] = 0.
        output[lower_energy_bound >= self._upper_energy] = 0.

        return output

    def total_flux_density(self):
        """
        Total flux density in units of
        [GeV cm^-2 s^-1]
        """

        norm = self._normalisation
        index = self._index
        lower, upper = self._lower_energy, self._upper_energy

        if index == 2:
            # special case
            int_norm = norm / (np.power(self._normalisation_energy, -index))
            return int_norm * (np.log(upper / lower))

        int_norm = norm / (np.power(self._normalisation_energy, -index) * (2 - index))
        return int_norm * (np.power(upper, 2 - index) - np.power(lower, 2 - index))

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        :param N: Number of samples.
        """

        return self.power_law.samples(N)

    def _rejection_sample(self, min_energy):
        """
        Sample energies from the power law.
        Uses rejection sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        """

        dist_upper_lim = self.spectrum(min_energy)

        accepted = False

        while not accepted:

            energy = np.random.uniform(min_energy, 1e3 * min_energy)
            dist = np.random.uniform(0, dist_upper_lim)

            if dist < self.spectrum(energy):

                accepted = True

        return energy

    def redshift_factor(self, z: float):

        return np.power(1 + z, 1 - self._index)


class BrokenPowerLawFlux(FluxModel):
    """
    Broken power law flux models.
    """

    def __init__(
        self,
        normalisation,
        break_energy,
        index1,
        index2,
        lower_energy=1e2,
        upper_energy=1e8,
    ):
        """
        Broken power law flux models.

        :param normalisation: Flux normalisation [GeV^-1 cm^-2 s^-1 sr^-1] or [GeV^-1 cm^-2 s^-1] for point sources.
        :param break_energy: Energy at which PL is broken and flux is normalised [GeV].
        :param index1: Index of the lower energy power law.
        :param index2: Index of the upper energy power law.
        :param lower_energy: Lower energy over which model is defined [GeV].
        :param upper_energy: Upper energy over which model is defined [GeV].
        """

        super().__init__()

        self._normalisation = normalisation

        self._break_energy = break_energy

        self._index1 = -index1

        self._index2 = -index2

        self._lower_energy = lower_energy

        self._upper_energy = upper_energy

        self.power_law = BrokenBoundedPowerLaw(
            lower_energy, break_energy, upper_energy, -index1, -index2
        )

    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO depending on flux_type.
        """

        if isinstance(energy, np.ndarray):
            norm = self._normalisation
            nans = np.nonzero(((energy < self._lower_energy) | (energy > self._upper_energy)))

            output = np.zeros_like(energy)

            below = np.nonzero((energy < self._break_energy))
            output[below] = norm * np.power(energy[below] / self._break_energy, self._index1)

            above = np.nonzero((energy > self._break_energy))
            output[above] = norm * np.power(energy[above] / self._break_energy, self._index2)

            middle = np.nonzero((energy == self._break_energy))
            output[middle] = norm

            output[nans] = 0

            return output

        else:
            if (energy < self._lower_energy) or (energy > self._upper_energy):

                return 0

            else:

                norm = self._normalisation

                if energy < self._break_energy:

                    output = norm * np.power(energy / self._break_energy, self._index1)

                elif energy == self._break_energy:

                    output = norm

                elif energy > self._break_energy:

                    output = norm * np.power(energy / self._break_energy, self._index2)

                return output

    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        \int spectrum dE over finite energy bounds.

        :param lower_energy_bound: [GeV]
        :param upper_energy_bound: [GeV]
        """
        if isinstance(lower_energy_bound, np.ndarray) and isinstance(upper_energy_bound, np.ndarray):
            norm = self._normalisation
            E_break = self._break_energy
            g1 = self._index1
            g2 = self._index2

            val = np.zeros_like(lower_energy_bound)

            idx1 = np.logical_and(lower_energy_bound < E_break, upper_energy_bound <= E_break)
            idx2 = np.logical_and(lower_energy_bound < E_break, upper_energy_bound > E_break)
            idx3 = np.logical_and(lower_energy_bound >= E_break, upper_energy_bound > E_break)

            val[idx1] = norm * (np.power(upper_energy_bound[idx1], g1 + 1.0) - np.power(lower_energy_bound[idx1], g1 + 1.0))/ ((g1 + 1.0) * np.power(E_break, g1))
            val[idx2] = norm * ((np.power(E_break, g1 + 1.0) - np.power(lower_energy_bound[idx2], g1 + 1.0)) / ((g1 + 1.0) * np.power(E_break, g1)) + (np.power(upper_energy_bound[idx2], g2 + 1.0) - np.power(E_break, g2 + 1.0)) / ((g2 + 1.0) * np.power(E_break, g2)))
            val[idx3] = norm * (np.power(upper_energy_bound[idx3], g2 + 1.0) - np.power(lower_energy_bound[idx3], g2 + 1.0)) / ((g2 + 1.0) * np.power(E_break, g2))

            return val

        else:
            norm = self._normalisation

            if (lower_energy_bound < self._break_energy) and (
                upper_energy_bound <= self._break_energy
            ):

                # Integrate over lower segment
                output = (
                    norm
                    * (
                        np.power(upper_energy_bound, self._index1 + 1.0)
                        - np.power(lower_energy_bound, self._index1 + 1.0)
                    )
                    / ((self._index1 + 1.0) * np.power(self._break_energy, self._index1))
                )

            elif (lower_energy_bound < self._break_energy) and (
                upper_energy_bound > self._break_energy
            ):

                # Integrate across break
                lower = (
                    np.power(self._break_energy, self._index1 + 1.0)
                    - np.power(lower_energy_bound, self._index1 + 1.0)
                ) / ((self._index1 + 1.0) * np.power(self._break_energy, self._index1))

                upper = (
                    np.power(upper_energy_bound, self._index2 + 1.0)
                    - np.power(self._break_energy, self._index2 + 1.0)
                ) / ((self._index2 + 1.0) * np.power(self._break_energy, self._index2))

                output = norm * (lower + upper)

            elif (lower_energy_bound >= self._break_energy) and (
                upper_energy_bound > self._break_energy
            ):

                # Integrate over upper segment
                upper = (
                    np.power(upper_energy_bound, self._index2 + 1.0)
                    - np.power(lower_energy_bound, self._index2 + 1.0)
                ) / ((self._index2 + 1.0) * np.power(self._break_energy, self._index2))

                output = norm * upper

            return output
        

    def redshift_factor(self, z: float):
        return 1.0

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param N: Number of samples.
        """

        return self.power_law.samples(N)


class PowerLawExpCutoffFlux(FluxModel):
    """
    Power law flux models with an exponential cutoff.
    """

    def __init__(
            self,
            normalisation,
            norm_energy,
            index,
            cutoff_energy,
            lower_energy=1e2,
            upper_energy=1e8
    ):
        """
        Power law flux models with an exponential cutoff.

        :param normalisation: Flux normalisation [GeV^-1 cm^-2 s^-1 sr^-1] or [GeV^-1 cm^-2 s^-1] for point sources.
        :param norm_energy: Energy at which the spectrum is normalised [GeV].
        :param index: Spectral index of the power law.
        :param cutoff_energy: Cutoff energy [GeV].
        :param lower_energy: Lower energy bound [GeV].
        :param upper_energy: Upper energy bound [GeV].
        """

        super().__init__()

        self._normalisation = normalisation

        self._norm_energy = norm_energy

        self._index = index

        self._cutoff_energy = cutoff_energy

        self._lower_energy = lower_energy

        self._upper_energy = upper_energy

        self.power_law = BoundedPowerLawExpCutoff(self._index, self._cutoff_energy, self._lower_energy, self._upper_energy)

    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO.
        """
        output = self._normalisation * (energy/self._norm_energy)**(-self._index) * np.exp(-energy/self._cutoff_energy)

        if isinstance(energy, np.ndarray):
            idx = np.logical_or(energy < self._lower_energy, energy > self._upper_energy)
            output[idx] = np.zeros(len(output[idx]))
            return output

        else:
            if energy < self._lower_energy or energy > self._upper_energy:
                return 0.0
            else:
                return output

    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        Integrates the spectrum with respect to E over a finite energy interval.
        :param lower_energy_bound: in GeV
        :param upper_energy_bound: in GeV
        """
        norm = self._normalisation
        E0 = self._norm_energy
        Ecut = self._cutoff_energy
        gam = self._index

        E1 = lower_energy_bound
        E2 = upper_energy_bound
        incGamma = np.frompyfunc(mp.gammainc, 3, 1)

        # Emin = self._lower_energy
        # Emax = self._upper_energy
        # if E1 <= Emin and E2 <= Emin:
        #     return 0.0
        # elif E1 <= Emin and E2 <= Emax:
        #     return norm * Ecut**(1-gam) * float(incGamma(1-gam, Emin/Ecut, E2/Ecut))
        # elif E1 > Emin and E2 <= Emax:
        #     return norm * Ecut**(1-gam) * float(incGamma(1-gam, E1/Ecut, E2/Ecut))
        # elif E1 > Emin and E2 > Emax:
        #     return norm * Ecut**(1-gam) * float(incGamma(1-gam, E1/Ecut, Emax/Ecut))
        # elif E1 <= Emin and E2 > Emax:
        #     return norm * Ecut**(1-gam) * float(incGamma(1-gam, Emin/Ecut, Emax/Ecut))
        # else:
        #     return 0.0

        if isinstance(lower_energy_bound, np.ndarray) or isinstance(upper_energy_bound, np.ndarray):
            return norm/E0**(-gam) * Ecut**(1-gam) * incGamma(1-gam, E1/Ecut, E2/Ecut).astype('float64')

        else:
            return norm/E0**(-gam) * Ecut**(1-gam) * float(incGamma(1-gam, E1/Ecut, E2/Ecut))

    def redshift_factor(self, z: float):
        return 1.0

    def sample(self, N):
        """
        Samples energies from the spectrum using inverse transform sampling.
        Works only if index < 1.
        :param N: Number of samples.
        """
        return self.power_law.samples(N)
    

class PowerLawSubexpCutoffFlux(FluxModel):
    """
    Power law flux models with a subexponential cutoff.
    """

    def __init__(
            self,
            normalisation,
            norm_energy,
            index1,
            cutoff_energy,
            index2,
            lower_energy=1e2,
            upper_energy=1e8
    ):
        """
        Power law flux models with a subexponential cutoff.

        :param normalisation: Flux normalisation [GeV^-1 cm^-2 s^-1 sr^-1] or [GeV^-1 cm^-2 s^-1] for point sources.
        :param norm_energy: Energy at which the spectrum is normalised [GeV].
        :param index1: Spectral index of the power law.
        :param cutoff_energy: Cutoff energy [GeV].
        :param index2: Spectral index in the exponential function. 0 < index2 < 1.
        :param lower_energy: Lower energy bound [GeV].
        :param upper_energy: Upper energy bound [GeV].
        """

        super().__init__()

        self._normalisation = normalisation

        self._norm_energy = norm_energy

        self._index1 = index1

        self._cutoff_energy = cutoff_energy

        self._index2 = index2

        self._lower_energy = lower_energy

        self._upper_energy = upper_energy

        self.power_law = BoundedPowerLawSubexpCutoff(self._index1, self._cutoff_energy, self._index2, self._lower_energy, self._upper_energy)

    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO.
        """
        output = self._normalisation * (energy/self._norm_energy)**(-self._index1) * np.exp(-1 * (energy/self._cutoff_energy)**self._index2)

        if isinstance(energy, np.ndarray):
            idx = np.logical_or(energy < self._lower_energy, energy > self._upper_energy)
            output[idx] = np.zeros(len(output[idx]))
            return output
        
        else:
            if energy < self._lower_energy or energy > self._upper_energy:
                return 0.0
            else:
                return output
            
    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        Integrates the spectrum with respect to E over a finite energy interval.
        :param lower_energy_bound: in GeV
        :param upper_energy_bound: in GeV
        """
        norm = self._normalisation
        E0 = self._norm_energy
        Ecut = self._cutoff_energy
        gamma = self._index1
        lambda_ = self._index2

        E1 = lower_energy_bound
        E2 = upper_energy_bound
        incGamma = np.frompyfunc(mp.gammainc, 3, 1)

        if isinstance(lower_energy_bound, np.ndarray) or isinstance(upper_energy_bound, np.ndarray):
            return norm/E0**(-gamma) * Ecut**(1-gamma)/lambda_ * incGamma((1-gamma)/lambda_, (E1/Ecut)**lambda_, (E2/Ecut)**lambda_).astype('float64')
        
        else:
            return norm/E0**(-gamma) * Ecut**(1-gamma)/lambda_ * float(incGamma((1-gamma)/lambda_, (E1/Ecut)**lambda_, (E2/Ecut)**lambda_))
        
    def redshift_factor(self, z: float):
        return 1.0
import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType


class PulseShape:
    """
    Playing with pulse shaping functions
    """
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        np.random.seed(self.seed)
        self.A = np.random.uniform(0.0, 100.0, self.N)
        self.phi = np.random.uniform(-np.pi, np.pi, self.N)
        self.frequency = np.linspace(
            self.freq_center - self.freq_width,
            self.freq_center + self.freq_width - 2.*self.freq_width/self.N,
            self.N
        )
        print self.frequency
        print self.A
        print self.phi
        self.t = np.linspace(self.tmin, self.tmax, self.N_tsteps)

    def A_P_pulse(self):
        """
        Amplitude-phase pulse shape  function: \sum A_n cos(w_n t + phi_n)
        """
        return np.exp(-(self.t-self.t_c)**2/self.sig2)*np.asarray([self.A[i]*(np.cos(self.frequency[i]*self.t + self.phi[i])) for i in range(self.N)])


if __name__ == '__main__':
    shape = PulseShape(
        seed=12,
        N=20,
        freq_center=300.,
        freq_width=50,
        tmin=0.,
        tmax=1.,
        t_c=0.5,
        sig2=0.025,
        N_tsteps=1000,
    )

    import seaborn as sns
    from scipy.optimize import curve_fit

    shape.seed = 121
    sns.set_palette(sns.hls_palette(8, l=.3, s=.8))
    pulse = shape.A_P_pulse()
    plt.figure()
    plt.plot(shape.t, pulse.sum(axis=0), 'k')
    plt.show()
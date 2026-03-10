class Neuron:

    def __init__(self, V_rest, V_reset, V_threshold, tau_m, R, I, w=0):
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_threshold = V_threshold
        self.tau_m = tau_m
        self.R = R
        self.I = I
        self.weight = w
        self.dV = None
    
    def calculate(self, V, synaptic_input=0):
        dV = (1 / self.tau_m) * (-(V - self.V_rest) + self.R * (self.I + synaptic_input))
        return dV
    
'''
V_rest = mV
V_reset = mV
V_threshold = mV
tau_m = ms
R = M Ohms
I = nA
'''
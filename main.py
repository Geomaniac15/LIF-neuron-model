'''
1. Start at V = V_{rest}
2. Each timestep, compute dV/dt using the equation
3. Update V += dV ⋅ dt
4. If V ≥ V_{threshold}, record a spike and reset V = V_{reset}
5. Repeat
'''

V_rest = -70 # mV
V_threshold = -55 # mV
tau_m = 20 # ms
R = 10 # M Ohms
I = 2 # nA
dt = 1 # ms

def calculate(V, V_rest, R, I):
    dV = (1 / 20) * (-(V - V_rest) + R * I)
    return dV

print(V_rest + calculate(V_rest, V_rest, R, I))
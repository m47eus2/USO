import numpy as np
import matplotlib.pyplot as plt

# Punkt pracy 400

omega_400 = np.array([
    40.0, 40.4, 40.5, 40.7, 40.9, 41.1, 41.3,
    41.5, 41.7, 41.9, 42.0, 42.1, 42.3, 42.4,
    42.5, 42.7, 42.8, 43.1, 43.1, 43.2, 43.3
])

M_400 = np.array([
     4.945815,  4.451234,  3.956652,  3.462071,  2.967489,
     2.472908,  1.978326,  1.483745,  0.989163,  0.494582,
     0.0,       -0.49458,  -0.98916,  -1.48374,  -1.97833,
    -2.47291,  -2.96749,  -3.46207,  -3.95665,  -4.45123,
    -4.94582
])

# Punkt pracy 800

omega_800 = np.array([
    82.0, 82.3, 82.5, 82.6, 82.9, 83.0, 83.2,
    83.3, 83.4, 83.7, 83.8, 84.0, 84.1, 84.2,
    84.4, 84.5, 84.6, 84.8, 84.9, 85.1, 85.2
])

M_800 = np.array([
     4.945815,  4.451234,  3.956652,  3.462071,  2.967489,
     2.472908,  1.978326,  1.483745,  0.989163,  0.494582,
     0.0,       -0.49458,  -0.98916,  -1.48374,  -1.97833,
    -2.47291,  -2.96749,  -3.46207,  -3.95665,  -4.45123,
    -4.94582
])

# Punkt pracy 1200

omega_1200 = np.array([
    124.0, 124.3, 124.4, 124.6, 124.7, 124.9, 125.1,
    125.2, 125.4, 125.6, 125.7, 125.8, 126.0, 126.1,
    126.3, 126.4, 126.5, 126.7, 126.7, 126.9, 127.0
])

M_1200 = np.array([
     4.945815,  4.451234,  3.956652,  3.462071,  2.967489,
     2.472908,  1.978326,  1.483745,  0.989163,  0.494582,
     0.0,       -0.49458,  -0.98916,  -1.48374,  -1.97833,
    -2.47291,  -2.96749,  -3.46207,  -3.95665,  -4.45123,
    -4.94582
])

# Regresja liniowa

a_400,b_400 = np.polyfit(omega_400, M_400, 1)
M_fit_400 = a_400 * omega_400 + b_400

a_800,b_800 = np.polyfit(omega_800, M_800, 1)
M_fit_800 = a_800 * omega_800 + b_800

a_1200,b_1200 = np.polyfit(omega_1200, M_1200, 1)
M_fit_1200 = a_1200 * omega_1200 + b_1200

# Wykresy

plt.figure()
plt.title('Charakterystyka M(ω) - Punkt pracy 400 obr/min')
plt.xlabel('ω [rad/s]')
plt.ylabel('M [Nm]')
plt.grid(True)

plt.plot(omega_400, M_400, 'o-', label="Pomiary punkt pracy = 400 obr/min")
plt.plot(omega_400, M_fit_400, '--', label=f'Regresja: M = {a_400:.3f}·ω + {b_400:.3f}')

plt.legend()
plt.tight_layout()

plt.figure()
plt.title('Charakterystyka M(ω) - Punkt pracy 800 obr/min')
plt.xlabel('ω [rad/s]')
plt.ylabel('M [Nm]')
plt.grid(True)

plt.plot(omega_800, M_800, 'o-', label="Pomiary punkt pracy = 800 obr/min")
plt.plot(omega_800, M_fit_800, "--", label=f"Regresja: M = {a_800:.3f}·ω + {b_800:.3f}")

plt.legend()
plt.tight_layout()

plt.figure()
plt.title('Charakterystyka M(ω) - Punkt pracy 1200 obr/min')
plt.xlabel('ω [rad/s]')
plt.ylabel('M [Nm]')
plt.grid(True)

plt.plot(omega_1200, M_1200, 'o-', label="Pomiary punkt pracy = 1200 obr/min")
plt.plot(omega_1200, M_fit_1200, "--", label=f"Regresja: M = {a_1200:.3f}·ω + {b_1200:.3f}")

plt.legend()
plt.tight_layout()

plt.show()
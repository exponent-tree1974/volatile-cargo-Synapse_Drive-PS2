import numpy as np
import pandas as pd

# =========================
# Physical parameters
# =========================
m_s = 290.0
m_u = 59.0
k_s = 16000.0
k_t = 190000.0

c_min = 800.0
c_max = 3500.0

# =========================
# Load dataset
# =========================
df = pd.read_csv("road_profiles.csv")
t = df["t"].values
dt = t[1] - t[0]

road_profiles = {f"profile_{i}": df[f"profile_{i}"].values for i in range(1, 6)}

# =========================
# Smooth Skyhook Controller
# =========================
def smooth_skyhook_controller(v_s, v_u, c_min, c_max):
    # Deadband: mid damping (KEEP)
    if abs(v_s) < 0.008:
        return 0.5 * (c_min + c_max)

    rel_v = v_s - v_u
    s = v_s * rel_v

    # Slightly sharper than 4.2
    gain = np.tanh(4.28 * s)

    # Slightly reduced damping span (5% inward)
    c_lo = c_min + 0.05 * (c_max - c_min)
    c_hi = c_max - 0.05 * (c_max - c_min)

    c = 0.5 * (c_hi + c_lo) + 0.5 * (c_hi - c_lo) * gain
    return np.clip(c, c_min, c_max)







# =========================
# Simulation
# =========================
def simulate_profile(r, params):
    m_s, m_u, k_s, k_t, c_min, c_max, dt = params
    N = len(r)

    z_s = np.zeros(N)
    z_u = np.zeros(N)
    v_s = np.zeros(N)
    v_u = np.zeros(N)
    a_s = np.zeros(N)

    # 20 ms actuator delay = 4 steps
    c_buffer = [0.5 * (c_min + c_max)] * 4

    for i in range(N - 1):
        c_req = smooth_skyhook_controller(
            v_s[i], v_u[i],
            c_min, c_max
        )

        # Apply delay
        c_buffer.append(c_req)
        c = c_buffer.pop(0)

        # Dynamics
        a_s_i = (k_s * (z_u[i] - z_s[i]) +
                 c * (v_u[i] - v_s[i])) / m_s

        a_u_i = (-k_s * (z_u[i] - z_s[i]) -
                 c * (v_u[i] - v_s[i]) +
                 k_t * (r[i] - z_u[i])) / m_u

        # Semi-implicit Euler integration
        v_s[i + 1] = v_s[i] + a_s_i * dt
        v_u[i + 1] = v_u[i] + a_u_i * dt

        z_s[i + 1] = z_s[i] + v_s[i + 1] * dt
        z_u[i + 1] = z_u[i] + v_u[i + 1] * dt

        a_s[i + 1] = a_s_i

    return z_s, z_u, a_s

# =========================
# Metrics
# =========================
def compute_metrics(z_s, a_s, dt):
    z_s_rel = z_s - z_s[0]

    rms_zs = np.sqrt(np.mean(z_s_rel**2))
    max_zs = np.max(np.abs(z_s_rel))

    jerk = np.diff(a_s) / dt
    rms_jerk = np.sqrt(np.mean(jerk**2))
    jerk_max = np.max(np.abs(jerk))

    comfort_score = (
        0.5 * rms_zs +
        max_zs +
        0.5 * rms_jerk +
        jerk_max
    )

    return rms_zs, max_zs, rms_jerk, comfort_score

# =========================
# Run all profiles
# =========================
params = (m_s, m_u, k_s, k_t, c_min, c_max, dt)
results = []

for name, road in road_profiles.items():
    z_s, z_u, a_s = simulate_profile(road, params)
    rms_zs, max_zs, rms_jerk, comfort = compute_metrics(z_s, a_s, dt)

    results.append({
        "profile": name,
        "rms_zs": rms_zs,
        "max_zs": max_zs,
        "rms_jerk": rms_jerk,
        "comfort_score": comfort
    })

# =========================
# Write submission
# =========================
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)

print("submission.csv generated successfully")
print(submission_df)

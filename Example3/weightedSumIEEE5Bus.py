import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display

def weightedSumIEEE5Bus():
    start_time = time.time()
    np.random.seed(0)

    # --- Counters ---
    counts = {"f1": 0, "f2": 0, "f3": 0, "cons": 0, "obj": 0}

    # =========================================================================
    # 1. System Data Definitions (IEEE 5-Bus)
    # =========================================================================
    baseMVA = 100.0
    nb = 5
    ng = 5

    bus_data = np.array([
        [1, 3, 0.0],
        [2, 2, 0.0],
        [3, 1, 300.0],
        [4, 3, 300.0],
        [5, 1, 400.0]
    ])
    Pd = bus_data[:, 2]

    gen_params = [
        {'bus': 1, 'bg': 14.0, 'pmin': 0.0, 'pmax': 40.0},
        {'bus': 2, 'bg': 15.0, 'pmin': 0.0, 'pmax': 170.0},
        {'bus': 3, 'bg': 30.0, 'pmin': 0.0, 'pmax': 520.0},
        {'bus': 4, 'bg': 40.0, 'pmin': 0.0, 'pmax': 200.0},
        {'bus': 5, 'bg': 10.0, 'pmin': 0.0, 'pmax': 600.0},
    ]

    branch_data = [
        (1, 2, 0.0281, 400.0),
        (1, 4, 0.0304, 9999.0),
        (1, 5, 0.0064, 9999.0),
        (2, 3, 0.0108, 9999.0),
        (3, 4, 0.0297, 9999.0),
        (4, 5, 0.0297, 240.0),
    ]

    Pg_ref = np.array([40.0, 170.0, 323.4948, 0.0, 466.5052])

    # =========================================================================
    # 2. Build Network Matrices (DC approximation)
    # =========================================================================
    Bbus = np.zeros((nb, nb))
    Bf = np.zeros((len(branch_data), nb))

    for idx, (f_bus, t_bus, x, limit) in enumerate(branch_data):
        i, j = f_bus - 1, t_bus - 1
        b_ij = 1.0 / x
        Bbus[i, i] += b_ij
        Bbus[j, j] += b_ij
        Bbus[i, j] -= b_ij
        Bbus[j, i] -= b_ij
        Bf[idx, i] = b_ij
        Bf[idx, j] = -b_ij

    # Generator-to-Bus Mapping
    Ag = np.zeros((nb, ng))
    for g_idx, gen in enumerate(gen_params):
        Ag[gen['bus']-1, g_idx] = 1.0

    print(f"Ag = {Ag}")
    # =========================================================================
    # 3. Define Objective Functions and Constraints
    # =========================================================================
    bg_vec = np.array([g['bg'] for g in gen_params])
    f_limits = np.array([b[3] for b in branch_data])
    pmin = np.array([g['pmin'] for g in gen_params])
    pmax = np.array([g['pmax'] for g in gen_params])

    # Decision vector: x = [Pg1,...Pg5, theta1,...theta5].
    # Bus 4 is the reference bus: theta4 = 0.
    def unpack_x(x):
        Pg = x[:ng]
        theta = x[ng:]
        # breakpoint()
        return Pg, theta

    # Objectives
    def f1_obj(x):
        counts["f1"] += 1
        Pg, _ = unpack_x(x)
        return np.sum(bg_vec * Pg)

    def f2_obj(x):
        counts["f2"] += 1
        Pg, _ = unpack_x(x)
        return np.sum((Pg - Pg_ref)**2)

    def f3_obj(x):
        counts["f3"] += 1
        flows = line_flows_selected(x)
        return np.sum((flows / line_limits)**2)

    # Equality constraints: nodal power balance
    def power_balance(x):
        counts["cons"] += 1
        Pg, theta = unpack_x(x)
        return Ag @ Pg - Pd - baseMVA*(Bbus @ theta)


    def line_flows_selected(x):
        _, theta = unpack_x(x)
        # Compute only the selected line flows (in MW)
        flow_1_2 = baseMVA * (Bf[0] @ theta)
        flow_4_5 = baseMVA * (Bf[5] @ theta)
        return np.array([flow_1_2, flow_4_5])

    line_limits = np.array([400, 240])

    def thermal_limits_upper(x):
        flows = line_flows_selected(x)
        return line_limits - flows   # flow <= limit

    def thermal_limits_lower(x):
        flows = line_flows_selected(x)
        return line_limits + flows   # flow >= -limit

    # def line_flows(x):
    #     _, theta = unpack_x(x)
    #     return baseMVA * (Bf @ theta)


    # def thermal_limits_upper(x):
    #     flows = line_flows(x)
    #     breakpoint()
    #     return f_limits - np.abs(flows)

    # def thermal_limits_lower(x):
    #     flows = line_flows(x)
    #     return f_limits + flows

    def ref_angle_constraint(x):
        _, theta = unpack_x(x)
        return theta[3]

    # Decision variables bounds
    bounds = [(g['pmin'], g['pmax']) for g in gen_params] + [(-np.pi, np.pi)]*nb

    # =========================================================================
    # 4. Weighted Sum Optimization Loop
    # =========================================================================
    weights = np.linspace(0,1,6)
    solutions = []
    x0 = np.zeros(ng + nb)  # initial guess: Pg zeros, theta zeros

    # weight_output = []
    # Wset = [
    #         # [0.00, 0.20, 0.80],
    #         # [0.00, 0.40, 0.60],
    #         # [0.00, 0.60, 0.40],
    #         # [0.00, 0.80, 0.20],
    #         # [0.00, 1.00, 0.00],
    #         # [0.20, 0.00, 0.80],
    #         [0.20, 0.20, 0.60],
    #         [0.20, 0.40, 0.40],
    #         [0.20, 0.60, 0.20],
    #         # [0.20, 0.80, 0.00],
    #         # [0.40, 0.00, 0.60],
    #         [0.40, 0.20, 0.40],
    #         [0.40, 0.40, 0.20],
    #         # [0.40, 0.60, 0.00],
    #         # [0.60, 0.00, 0.40],
    #         [0.60, 0.20, 0.20],
    #         # [0.60, 0.40, 0.00],
    #         # [0.80, 0.00, 0.20],
    #         # [0.80, 0.20, 0.00],
    #         # [1.00, 0.00, 0.00]
    #         ]

    Wset = [
        [0.075, 0.150, 0.775], [0.075, 0.225, 0.700],
        [0.075, 0.300, 0.625], [0.075, 0.375, 0.550], [0.075, 0.450, 0.475],
        [0.075, 0.525, 0.400], [0.075, 0.600, 0.325], [0.075, 0.675, 0.250],
        [0.075, 0.750, 0.175], [0.075, 0.825, 0.100], [0.150, 0.075, 0.775],
        [0.150, 0.150, 0.700], [0.150, 0.225, 0.625], [0.150, 0.300, 0.550],
        [0.150, 0.375, 0.475], [0.150, 0.450, 0.400], [0.150, 0.525, 0.325],
        [0.150, 0.600, 0.250], [0.150, 0.675, 0.175], [0.150, 0.750, 0.100],
        [0.225, 0.075, 0.700], [0.225, 0.150, 0.625], [0.225, 0.225, 0.550],
        [0.225, 0.300, 0.475], [0.225, 0.375, 0.400], [0.225, 0.450, 0.325],
        [0.225, 0.525, 0.250], [0.225, 0.600, 0.175], [0.225, 0.675, 0.100],
        [0.300, 0.075, 0.625], [0.300, 0.150, 0.550], [0.300, 0.225, 0.475],
        [0.300, 0.300, 0.400], [0.300, 0.375, 0.325], [0.300, 0.450, 0.250],
        [0.300, 0.525, 0.175], [0.300, 0.600, 0.100], [0.375, 0.075, 0.550],
        [0.375, 0.150, 0.475], [0.375, 0.225, 0.400], [0.375, 0.300, 0.325],
        [0.375, 0.375, 0.250], [0.375, 0.450, 0.175], [0.375, 0.525, 0.100],
        [0.450, 0.075, 0.475], [0.450, 0.150, 0.400], [0.450, 0.226, 0.324],
        [0.450, 0.301, 0.249], [0.450, 0.377, 0.173]
        ]

    for weight in Wset:
        w1, w2, w3 = weight

        def obj(x):
            counts["obj"] += 1
            return w1*f1_obj(x) + w2*f2_obj(x) + w3*f3_obj(x)

        # cons = [{'type':'eq', 'fun': power_balance}]
        cons = [
                {'type': 'eq',   'fun': power_balance},
                {'type': 'eq',   'fun': ref_angle_constraint},
                {'type': 'ineq', 'fun': thermal_limits_upper},
                {'type': 'ineq', 'fun': thermal_limits_lower},
            ]

        # Solve using SLSQP
        sol = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                        options={'ftol':1e-9, 'disp':False, 'maxiter':2000})

        if sol.success:
            Pg_sol, theta_sol = unpack_x(sol.x)
            flows_sol = baseMVA*(Bf @ theta_sol)
            solutions.append({
                'w1': w1, 'w2': w2, 'w3': w3,
                # 'solX': sol.x[3:],
                'f1': f1_obj(sol.x),
                'f2': f2_obj(sol.x),
                'f3': f3_obj(sol.x),
                'Pg': Pg_sol,
                'theta': theta_sol,
                # 'flows': flows_sol,
                'therm_lim_upp': -thermal_limits_upper(sol.x),
                'therm_lim_low': -thermal_limits_lower(sol.x),
                'h_pow': power_balance(sol.x),
                'h_ang': ref_angle_constraint(sol.x),
                'eq_max_MW': max(np.max(np.abs(power_balance(sol.x))), abs(ref_angle_constraint(sol.x))),
                'ineq_max_MW': max(0., np.max(-thermal_limits_upper(sol.x)), np.max(-thermal_limits_lower(sol.x)))
            })
        else:
            print(f"Solver failed for weights w1={w1}, w2={w2}, w3={w3}")

    df = pd.DataFrame(solutions)
    df.to_csv('ieee5bus_weighted_sum_figure13_corrected.csv', index=False)

    end_time = time.time()
    df = df.map(
        lambda x: [round(v, 3) for v in x] if isinstance(x, (list, tuple, np.ndarray)) else round(x, 3) if isinstance(x, float) else x
        )
    display("df solution", df)
    # display("power balance", df.h_power_balance)
    # display("flows sol", df.flows)

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print("Function evaluation counts:", counts)




    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: f1 vs f2
    sc1 = axes[0].scatter(df['f1'], df['f2'], c=df['w1'], cmap='viridis', s=50, alpha=0.7)
    axes[0].set_xlabel('Fuel Cost (f1)')
    axes[0].set_ylabel('Dispatch Deviation (f2)')
    axes[0].set_title('Pareto Front (f1 vs f2)')
    fig.colorbar(sc1, ax=axes[0], label='Weight w1')

    # Plot 2: f1 vs f3
    sc2 = axes[1].scatter(df['f1'], df['f3'], c=df['w1'], cmap='viridis', s=50, alpha=0.7)
    axes[1].set_xlabel('Fuel Cost (f1)')
    axes[1].set_ylabel('f3')
    axes[1].set_title('Pareto Front (f1 vs f3)')
    fig.colorbar(sc2, ax=axes[1], label='Weight w1')

    # Plot 3: f2 vs f3
    sc3 = axes[2].scatter(df['f2'], df['f3'], c=df['w1'], cmap='viridis', s=50, alpha=0.7)
    axes[2].set_xlabel('Dispatch Deviation (f2)')
    axes[2].set_ylabel('f3')
    axes[2].set_title('Pareto Front (f2 vs f3)')

    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    # fig.colorbar(sc, cax=cbar_ax, label='Weight w1')
    fig.colorbar(sc3, ax=axes[2], label='Weight w1')

    plt.tight_layout()
    plt.savefig('pareto_plots_weighted_sum_figure13_corrected.png')
    plt.show()



    # 3D Plotting logic
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(df['f1'], df['f2'], df['f3'], c=df['w1'], cmap='viridis', s=50, alpha=0.8)

    ax.set_xlabel('Fuel Cost (f1)')
    ax.set_ylabel('Dispatch Deviation (f2)')
    ax.set_zlabel('f3')
    ax.set_title('3D Pareto Front')

    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Weight w1')

    plt.tight_layout()
    plt.savefig('pareto_plot_3d_figure13_corrected.png')
    plt.show()



if __name__ == "__main__":
    weightedSumIEEE5Bus()

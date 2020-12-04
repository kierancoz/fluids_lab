import json
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def main():
    lab_data = json.load(open("lab_data.json","r+"))


    # Re = density * velocity * width / viscosity
    reynold_number = lambda velocity, width: 1.225 * velocity * width / (1.81 * 10**-5)
    # C_l/C_d = force / (area normal to flow * dynamic pressure)
    force_coefficient = lambda force, area, dyn_pres: force / (area * dyn_pres)
    # Pdyn = 1/2 * p * v^2
    p_to_vel = lambda x: sqrt(x * 2 / 1.225) if x > 0 else 0


    # # # # # # # # # # # # # #
    # # # 24" WIND TUNNEL # # #
    # # # # # # # # # # # # # #
    v_to_p_big = v_to_p_func(lab_data.get("calibration_large"), plot = False)
    v_to_drag_big = v_to_f_func(lab_data.get("calibration_large_drag"), plot = False)
    v_to_lift_big = v_to_f_func(lab_data.get("calibration_large_lift"), plot = False)

    # get large sting information
    sting_drag, sting_lift = get_forces(lab_data.get("large_sting"), v_to_drag_big, v_to_lift_big)
    sting_pressure = get_pressure(lab_data.get("large_sting"), v_to_p_big)
    sting_wind_velocities = [p_to_vel(x) for x in sting_pressure]

    # get forces for f1 car
    f1_total_drag, f1_total_lift = get_forces(lab_data.get("large_f1"), v_to_drag_big, v_to_lift_big)
    f1_drag = np.array(f1_total_drag) - np.array(sting_drag)
    f1_lift = np.array(f1_total_lift) - np.array(sting_lift)

    # get forces for f150 gate up
    f150up_total_drag, f150up_total_lift = get_forces(lab_data.get("large_f150_up"), v_to_drag_big, v_to_lift_big)
    f150up_drag = np.array(f150up_total_drag) - np.array(sting_drag)
    f150up_lift = np.array(f150up_total_lift) - np.array(sting_lift)

    # get forces for f150 gate down
    f150down_total_drag, f150down_total_lift = get_forces(lab_data.get("large_f150_down"), v_to_drag_big, v_to_lift_big)
    f150down_drag = np.array(f150down_total_drag) - np.array(sting_drag)
    f150down_lift = np.array(f150down_total_lift) - np.array(sting_lift)

    # drag and lift plots for f1 and f150 in different configs
    plot_forces("Drag", sting_wind_velocities, plot = False, f1 = f1_drag, f150_up = f150up_drag, f150_down = f150down_drag)
    plot_forces("Lift", sting_wind_velocities, plot = False, f1 = f1_lift, f150_up = f150up_lift, f150_down = f150down_lift)
    
    # reynold numbers
    f1_rn = [reynold_number(vel, 0.1) for vel in sting_wind_velocities]
    f150_rn = [reynold_number(vel, 0.12065) for vel in sting_wind_velocities]

    # c_l calcs
    f1_c_l = [force_coefficient(x, 0.0042, sting_pressure[i]) for i, x in enumerate(f1_lift)]
    f150down_c_l = [force_coefficient(x, 0.00992, sting_pressure[i]) for i, x in enumerate(f150down_lift)]
    f150up_c_l = [force_coefficient(x, 0.00992, sting_pressure[i]) for i, x in enumerate(f150up_lift)]
    
    plot_force_coeff("C_l", plot = False, f1 = {"x" : f1_rn, "y" : f1_c_l}, f150_down = {"x" : f150_rn, "y" : f150down_c_l},
        f150_up = {"x" : f150_rn, "y" : f150up_c_l})

    # c_d calcs
    f1_c_d = [force_coefficient(x, 0.0042, sting_pressure[i]) for i, x in enumerate(f1_drag)]
    f150down_c_d = [force_coefficient(x, 0.00992, sting_pressure[i]) for i, x in enumerate(f150down_drag)]
    f150up_c_d = [force_coefficient(x, 0.00992, sting_pressure[i]) for i, x in enumerate(f150up_drag)]
    
    plot_force_coeff("C_d", plot = False, f1 = {"x" : f1_rn, "y" : f1_c_d}, f150_down = {"x" : f150_rn, "y" : f150down_c_d},
        f150_up = {"x" : f150_rn, "y" : f150up_c_d})


    # # # # # # # # # # # # # #
    # # # 12" WIND TUNNEL # # #
    # # # # # # # # # # # # # #
    v_to_p_small = v_to_p_func(lab_data.get("calibration_small"), plot = False)
    v_to_drag_small = v_to_f_func(lab_data.get("calibration_small_drag"), plot = False)

    # get sting information
    sting_drag_func = lambda velocity: 0.0007 * velocity ** 2 - 0.0028 * velocity + 0.0544 # from lab handout
    sting_pressure = get_pressure(lab_data.get("small_camaro_alone"), v_to_p_small)
    sting_wind_velocities = [p_to_vel(x) for x in sting_pressure]
    sting_drag = [sting_drag_func(x) for x in sting_wind_velocities]
    sting_lift = None # just so it doesnt accidentally get used anywhere, resetting this value

    # get camaro alone drag
    total_camaro_a_drag = get_forces(lab_data.get("small_camaro_alone"), v_to_drag_small)
    camaro_a_drag = np.array(total_camaro_a_drag) - np.array(sting_drag)

    # get drag for usps in front
    total_camaro_f_drag = get_forces(lab_data.get("small_usps_front"), v_to_drag_small)
    camaro_f_drag = np.array(total_camaro_f_drag) - np.array(sting_drag)

    # get drag for usps behind
    total_camaro_b_drag = get_forces(lab_data.get("small_usps_behind"), v_to_drag_small)
    camaro_b_drag = np.array(total_camaro_b_drag) - np.array(sting_drag)

    # zero drag forces
    zero_forces = lambda forces, zero : [x - zero for x in forces]
    camaro_a_drag = zero_forces(camaro_a_drag, camaro_a_drag[0])
    camaro_f_drag = zero_forces(camaro_f_drag, camaro_f_drag[0])
    camaro_b_drag = zero_forces(camaro_b_drag, camaro_b_drag[0])

    # drag plots for camaro with different USPS positions
    # plot NOT correct rn - need to account for Dstring using provided equation
    plot_forces("Drag", sting_wind_velocities, plot = True, just_camaro = camaro_a_drag, usps_front = camaro_f_drag, usps_behind = camaro_b_drag)

    # reynolds number
    camaro_rn = [reynold_number(vel, 0.0554) for vel in sting_wind_velocities]

    # c_d calcs
    justcamaro_c_d = [force_coefficient(x, 0.00206, sting_pressure[i]) for i, x in enumerate(camaro_a_drag)]
    uspsfront_c_d = [force_coefficient(x, 0.00206, sting_pressure[i]) for i, x in enumerate(camaro_f_drag)]
    uspsbehind_c_d = [force_coefficient(x, 0.00206, sting_pressure[i]) for i, x in enumerate(camaro_b_drag)]
    
    # somethings wrong with this plot lol
    plot_force_coeff("C_d", plot = True, just_camaro = {"x" : camaro_rn, "y" : justcamaro_c_d}, usps_front = {"x" : camaro_rn, "y" : uspsfront_c_d},
        usps_behind = {"x" : camaro_rn, "y" : uspsbehind_c_d})



def plot_force_coeff(ylabel, plot = True, **data):
    if plot:
        for x in data.values():
            plt.plot(x.get("x"), x.get("y"))
        plt.legend(data.keys())
        plt.ylabel(ylabel)
        plt.xlabel("Reynolds Number")
        plt.title("{0} vs Reynolds Number".format(ylabel))
        plt.show()

def plot_forces(ylabel, x_data, plot = True, **y_data):
    if plot:
        for y in y_data.values():
            plt.plot(x_data, y)
        plt.legend(y_data.keys())
        plt.ylabel("Force (N)")
        plt.xlabel("Velocity (m/s)")
        plt.title("Force from {0} vs Wind Velocity".format(ylabel))
        plt.show()

def get_pressure(data, v_to_p_func):
    pressure_index = data.get("categories").index("dynamic pressure")
    pressure_voltages = [(key, val[pressure_index]) for key, val in data.items() if key != "categories"]
    return [v_to_p_func(x[1]) for x in pressure_voltages]

def get_forces(data, drag_v_to_f, lift_v_to_f = None):
    drag_index = data.get("categories").index("drag")
    total_drag_voltages = [(key, val[drag_index]) for key, val in data.items() if key != "categories"]
    total_drag_forces = [drag_v_to_f(x[1]) for x in total_drag_voltages]

    # if there isnt lift, return drag only
    if not lift_v_to_f:
        return total_drag_forces
    
    lift_index = data.get("categories").index("lift")
    total_lift_voltages = [(key, val[lift_index]) for key, val in data.items() if key != "categories"]
    total_lift_forces = [lift_v_to_f(x[1]) for x in total_lift_voltages]

    # zero lift forces
    lift_force_zero = total_lift_forces[0]
    total_lift_forces = [x - lift_force_zero for x in total_lift_forces]

    return total_drag_forces, total_lift_forces

# linear fit of calibration data
def v_to_p_func(data, plot = False):
    # converts inH2O to pascals
    inH2O_to_Pa = lambda pressure_list: [x * 248.84 for x in pressure_list]

    x = [float(x) for x in list(data.keys())]
    y = inH2O_to_Pa(list(data.values()))
    z = np.polyfit(x,y,1)
    #print("Fit coefficients :", z)
    voltage_to_pressure_func = np.poly1d(z) 

    if plot:
        plt.plot(x,y, 'yo', x, voltage_to_pressure_func(x), '--k')
        plt.ylabel("Pressure Reading (Pa)")
        plt.xlabel("Voltage Reading (V)")
        plt.title("Pressure vs Voltage: P = {0} * V + {1}".format(int(z[0]), int(z[1])))
        plt.show()

    return voltage_to_pressure_func

# linear fit of calibration data
def v_to_f_func(data, plot = False):
    # converts kg to newtons
    kg_to_N = lambda mass_list: [float(x) * 9.81 for x in mass_list]

    y = kg_to_N(data.keys())
    x = [float(x) for x in data.values()]
    z = np.polyfit(x,y,1)
    #print("Fit coefficients :", z)
    voltage_to_force_func = np.poly1d(z) 

    if plot:
        plt.plot(x,y, 'yo', x, voltage_to_force_func(x), '--k')
        plt.ylabel("Force Reading (N)")
        plt.xlabel("Voltage Reading (V)")
        plt.title("Force vs Voltage: P = {0} * V + {1}".format(int(z[0]), int(z[1])))
        plt.show()

    return voltage_to_force_func

if __name__ == "__main__":
    main()
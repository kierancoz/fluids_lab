import numpy as np
import matplotlib.pyplot as plt
import json
from math import cos, pi, radians, sqrt

# linear fit of calibration data
def step_1(plot = False):
    data = lab_data.get("manometer")
    x = data.get("readings")
    y = data.get("volt_readout")
    coef = np.polyfit(x, y, 1)
    voltage_to_pressure_func = np.poly1d(coef) 

    if plot:
        plt.plot(x,y, 'yo', x, voltage_to_pressure_func(x), '--k')
        plt.ylabel("Pressure Reading (inH2O)")
        plt.xlabel("Voltage Reading (V)")
        plt.show()

    return voltage_to_pressure_func

# show pressure distribution around cylinder for 2 different velocities
def step_2(plot = False):
    slow_speed_data = [voltage_to_pressure_func(avg_data(a_list)) for a_list in lab_data.get("slow_speed_voltage")]
    fast_speed_data = [voltage_to_pressure_func(avg_data(a_list)) for a_list in lab_data.get("fast_speed_voltage")]
    theta = [i for i in range(0,190,15)]

    if plot:
        plt.plot(theta, slow_speed_data)
        plt.plot(theta, fast_speed_data)
        plt.ylabel("Pressure (inH2O)")
        plt.xlabel("Angle (theta)")
        plt.legend(["10 m/s", "20 m/s"])
        plt.title("Pressure Distribution")    
        plt.show()

    return slow_speed_data, fast_speed_data

# show Cp distribution around cylinder for 2 different velocities
def step_3(slow_pressure_list, fast_pressure_list, plot = False):
    up_p = lab_data.get("upstream_pressures")
    def cp(data, speed):
        # Cp = (surface pressure - upstream static pressure) / (upstream dynamic pressure)
        spd = up_p.get(speed)
        return (data - voltage_to_pressure_func(spd.get("voltage_stat"))) / voltage_to_pressure_func(spd.get("voltage_dyn"))

    slow_cp_data = [cp(x, "slow_speed") for x in slow_pressure_list]
    fast_cp_data = [cp(x, "fast_speed") for x in fast_pressure_list]
    theta = [i for i in range(0,190,15)]

    if plot:
        plt.plot(theta, slow_cp_data)
        plt.plot(theta, fast_cp_data)
        plt.ylabel("Cp")
        plt.xlabel("Angle (theta)")
        plt.legend(["10 m/s", "20 m/s"])
        plt.title("Pressure Distribution")
        plt.show()

# estimate total drag for both velocities
def step_4(slow_pressure_list, fast_pressure_list):
    cy_data = lab_data.get("cylinder_data")
    
    integral_func = lambda y, x : y * cos(radians(x))
    def integrate_pressure(pressure_list, step_size, start_val, end_val):
        integral_val = integrate(integral_func, pressure_list, start_val, end_val, step_size)
        return 2 * 2 * pi/360 * 1/2 * cy_data.get("length") * cy_data.get("diameter") * integral_val
    
    low_speed_integral = integrate_pressure(inH2O_to_Pa(slow_pressure_list), 15, 0, 180)
    fast_speed_integral = integrate_pressure(inH2O_to_Pa(fast_pressure_list), 15, 0, 180)
    return low_speed_integral, fast_speed_integral

# estimate Cd values
def step_5(slow_drag, fast_drag):
    slow_cd = cd(slow_drag, "slow_speed")
    fast_cd = cd(fast_drag, "fast_speed")
    return slow_cd, fast_cd

# find velocity profiles up & downstream and plot them against position
def step_6(plot = False):
    def dyn_pressure_to_velocity(pressure):
        # p = 1/2 * density * velocity ^ 2
        return sqrt(inH2O_to_Pa(pressure, is_list = False) * 2 / density)
    up_voltages = [avg_data(data) for data in lab_data.get("upstream_cv").get("p_dyn")]
    down_voltages = [avg_data(data) for data in lab_data.get("downstream_cv").get("p_dyn")]
    up_velocities = [dyn_pressure_to_velocity(voltage_to_pressure_func(volt)) for volt in up_voltages]
    down_velocities = [dyn_pressure_to_velocity(voltage_to_pressure_func(volt)) for volt in down_voltages]
    
    if plot:
        plt.plot(lab_data.get("upstream_cv").get("positions"), up_velocities)
        plt.plot(lab_data.get("downstream_cv").get("positions"), down_velocities)
        plt.ylabel("Velocity (m/s)")
        plt.xlabel("Position (in)")
        plt.legend(["Upstream", "Downstream"])
        plt.title("Velocity Profiles")
        plt.show()
    
    return up_velocities, down_velocities

# plot position against static pressure
def step_7(plot = False):
    up_voltages = [avg_data(data) for data in lab_data.get("upstream_cv").get("p_static")]
    down_voltages = [avg_data(data) for data in lab_data.get("downstream_cv").get("p_static")]
    up_pressures = [voltage_to_pressure_func(volt) for volt in up_voltages]
    down_pressures = [voltage_to_pressure_func(volt) for volt in down_voltages]

    if plot:
        plt.plot(lab_data.get("upstream_cv").get("positions"), up_pressures)
        plt.plot(lab_data.get("downstream_cv").get("positions"), down_pressures)
        plt.ylabel("Pressure (inH2O)")
        plt.xlabel("Position (in)")
        plt.legend(["Upstream", "Downstream"])
        plt.title("Static Pressure Profiles")
        plt.show()
    return up_pressures, down_pressures

# calculate Drag using control volume
def step_8(up_v, down_v, up_p, down_p):
    dyn_func = lambda y, x : density * lab_data.get("cylinder_data").get("length") * y**2
    upstream_dyn = integrate2(dyn_func, up_v, 0, len(up_v)-2, lab_data.get("upstream_cv").get("positions"))
    downstream_dyn = integrate2(dyn_func, down_v, 0, len(down_v)-2, lab_data.get("downstream_cv").get("positions"))
    
    stat_func = lambda y,x : y * lab_data.get("cylinder_data").get("length")
    up_p = inH2O_to_Pa(up_p)
    down_p = inH2O_to_Pa(down_p)
    upstream_stat = integrate2(stat_func, up_p, 0, len(up_p)-2, lab_data.get("upstream_cv").get("positions"))
    downstream_stat = integrate2(stat_func, down_p, 0, len(down_p)-2, lab_data.get("downstream_cv").get("positions"))

    return upstream_dyn - downstream_dyn + upstream_stat - downstream_stat

# # # helper functions # # #

# Cd = drag/(dynamic_pressure * Area)
def cd(drag, speed):
    dynamic_voltage = lab_data.get("upstream_pressures").get(speed).get("voltage_dyn")
    area = lab_data.get("cylinder_data").get("length") * lab_data.get("cylinder_data").get("diameter")/2 * pi
    return drag/inH2O_to_Pa(voltage_to_pressure_func(dynamic_voltage), is_list = False)/area

# converts inH2O to pascals
def inH2O_to_Pa(pressure, is_list = True):
    coefficient = 248.84
    if not is_list:
        return pressure*coefficient
    return [x*coefficient for x in pressure]

# euler integration (constant step size)
def integrate(func, input_list, start, end, step):
    total_integral = 0
    for i in range(int(start/step) + 1, int(end/step) + 1):
        partial_integral = func(input_list[i], step*i) * step
        total_integral += partial_integral
    return total_integral

# euler integration (variable step size)
def integrate2(func, input_list, start_indx, end_indx, step_list):
    total_integral = 0
    for i in range(start_indx + 1, end_indx):
        step_size = step_list[i] - step_list[i-1]
        partial_integral = func(input_list[i], step_list[i]) * step_size
        total_integral += partial_integral
    return total_integral

# average input data list
def avg_data(data):
    return sum(data)/len(data)

if __name__ == "__main__":
    global density
    density = 1.225 # air density

    global lab_data
    lab_data = json.load(open("lab_data.json","r+"))

    global voltage_to_pressure_func
    voltage_to_pressure_func = step_1(plot = False)

    slow_pressure_list, fast_pressure_list = step_2(plot = False)
    step_3(slow_pressure_list, fast_pressure_list, plot = False)
    slow_drag, fast_drag = step_4(slow_pressure_list, fast_pressure_list)
    print(slow_drag, fast_drag)
    slow_cd, fast_cd = step_5(slow_drag, fast_drag)
    print(slow_cd, fast_cd)
    up_v, down_v = step_6(plot = False)
    up_p, down_p = step_7(plot = False)
    drag = step_8(up_v, down_v, up_p, down_p)
    print(drag)
    cv_fast_cd = cd(drag, "fast_speed")
    print(cv_fast_cd)
    
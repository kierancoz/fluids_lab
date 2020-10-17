import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from math import cos, pi, radians, sqrt

# linear fit of calibration data
def step_1(plot = False):
    # converts inH2O to pascals
    inH2O_to_Pa = lambda pressure_list: [x * 248.84 for x in pressure_list]

    y = inH2O_to_Pa(lab_data.get("manometer").get("manometer_height"))
    x = lab_data.get("manometer").get("voltage_readout")
    voltage_to_pressure_func = np.poly1d(np.polyfit(x, y, 1)) 

    if plot:
        plt.plot(x,y, 'yo', x, voltage_to_pressure_func(x), '--k')
        plt.ylabel("Pressure Reading (Pa)")
        plt.xlabel("Voltage Reading (V)")
        plt.show()

    return voltage_to_pressure_func

# show pressure distribution around cylinder for 2 different velocities
def step_2(plot = False):
    slow_speed_data = [voltage_to_pressure_func(avg(a_list)) for a_list in lab_data.get("slow_speed_voltage")]
    fast_speed_data = [voltage_to_pressure_func(avg(a_list)) for a_list in lab_data.get("fast_speed_voltage")]
    theta = [i for i in range(0,190,15)]

    if plot:
        plt.plot(theta, slow_speed_data)
        plt.plot(theta, fast_speed_data)
        plt.ylabel("Pressure (Pa)")
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
        return (-data + voltage_to_pressure_func(spd.get("voltage_stat"))) / voltage_to_pressure_func(spd.get("voltage_dyn"))

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

# find velocity profiles up & downstream and plot them against position
def step_4(plot = False):
    # p = 1/2 * density * velocity ^ 2
    dyn_pressure_to_velocity = lambda pressure : sqrt(pressure * 2 / density)

    up_voltages = [avg(data) for data in lab_data.get("upstream_cv").get("p_dyn")]
    down_voltages = [avg(data) for data in lab_data.get("downstream_cv").get("p_dyn")]
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
def step_5(plot = False):
    up_voltages = [avg(data) for data in lab_data.get("upstream_cv").get("p_static")]
    down_voltages = [avg(data) for data in lab_data.get("downstream_cv").get("p_static")]
    up_pressures = [voltage_to_pressure_func(volt) for volt in up_voltages]
    down_pressures = [voltage_to_pressure_func(volt) for volt in down_voltages]

    if plot:
        plt.plot(lab_data.get("upstream_cv").get("positions"), up_pressures)
        plt.plot(lab_data.get("downstream_cv").get("positions"), down_pressures)
        plt.ylabel("Pressure (Pa)")
        plt.xlabel("Position (in)")
        plt.legend(["Upstream", "Downstream"])
        plt.title("Static Pressure Profiles")
        plt.show()
    return up_pressures, down_pressures

# calculate Drag using control volume
def step_7(up_v, down_v, up_p, down_p):
    with open('employee_file.csv', mode='w',newline="") as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        position_up = lab_data.get("upstream_cv").get("positions")
        position_down = lab_data.get("downstream_cv").get("positions")
        for i in range(len(down_v)):
            if i < len(up_v):
                employee_writer.writerow([position_up[i], up_v[i], up_p[i], "", position_down[i], down_v[i], down_p[i]])
            else:
                employee_writer.writerow(["", "", "", "", position_down[i], down_v[i], down_p[i]])

# # # helper functions # # #

# Cd = drag/(dynamic_pressure * Area)
def cd(drag, speed):
    dynamic_voltage = lab_data.get("upstream_pressures").get(speed).get("voltage_dyn")
    area = lab_data.get("cylinder_data").get("length") * lab_data.get("cylinder_data").get("diameter")
    print(voltage_to_pressure_func(dynamic_voltage))
    return drag / (voltage_to_pressure_func(dynamic_voltage) * area)

avg = lambda a_list: sum(a_list)/len(a_list)

# # # # # #

if __name__ == "__main__":
    global density
    density = 1.225 # air density

    global lab_data
    lab_data = json.load(open("lab_data.json","r+"))

    global voltage_to_pressure_func
    voltage_to_pressure_func = step_1(plot = True)

    # surface pressure analysis
    slow_pressure_list, fast_pressure_list = step_2(plot = True)
    step_3(slow_pressure_list, fast_pressure_list, plot = True)
    
    slow_drag, fast_drag = 0.415, 1.763 # get values from spreadsheet

    # control volume analysis
    slow_cd, fast_cd = cd(slow_drag, "slow_speed"), cd(fast_drag, "fast_speed")
    print(slow_cd, fast_cd)

    up_v, down_v = step_4(plot = True)
    up_p, down_p = step_5(plot = True)

    step_7(up_v, down_v, up_p, down_p)
    cv_drag = 1.3377 # get values from spreadsheet

    print(cv_drag)
    cv_fast_cd = cd(cv_drag, "fast_speed")
    print(cv_fast_cd)
    
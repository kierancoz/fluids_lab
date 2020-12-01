import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    lab_data = json.load(open("lab_data.json","r+"))

    v_to_p_small = v_to_p_func(lab_data.get("calibration_small"), plot = True)
    v_to_p_big = v_to_p_func(lab_data.get("calibration_big"), plot = True)
    print(v_to_p_big(1.749))

# linear fit of calibration data
def v_to_p_func(data, plot = False):
    # converts inH2O to pascals
    inH2O_to_Pa = lambda pressure_list: [float(x) * 248.84 for x in pressure_list]

    y = inH2O_to_Pa(data.keys())
    x = list(data.values())
    z = np.polyfit(x,y,1)
    print("Fit coefficients :", z)
    voltage_to_pressure_func = np.poly1d(z) 

    if plot:
        plt.plot(x,y, 'yo', x, voltage_to_pressure_func(x), '--k')
        plt.ylabel("Pressure Reading (Pa)")
        plt.xlabel("Voltage Reading (V)")
        plt.title("Pressure vs Voltage: P = {0} * V + {1}".format(int(z[0]), int(z[1])))
        plt.show()

    return voltage_to_pressure_func

if __name__ == "__main__":
    main()
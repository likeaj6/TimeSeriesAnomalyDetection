# import datetime
import sys
import numpy as np
from pandas.io.json import json_normalize
import pandas as pd
import json

def flattenjson(data, key):
    result = json_normalize(data, key)
    return result

def generatePlot():
    times = pd.date_range('2015-10-06', periods=96, freq='15min')

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()

    plt.plot(times, flat['value'])

    xfmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    # flat.plot()
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.show()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)
    return 0
def generateCSV(dataframe, output):
    result = dataframe.to_csv(output, encoding='utf-8')
    return result

def on_plot_hover(event):
    return 0

# key = 'dataset'
# filepath = 'testFitBitData.json'
# output = 'testFitBit.csv'



# fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "\nUsage: python json_csv.py [json filepath] [nodename_to_convert] [output filepath] [0 or 1 to plot data]\n"
    else:
        filepath = sys.argv[1]
        key = sys.argv[2]
        output = sys.argv[3]
        plot = sys.argv[4]

        with open(filepath) as json_data:
            d = json.load(json_data)
            # print(d["activities-steps-intraday"]["dataset"])
            flat = flattenjson(d["activities-steps-intraday"], key)
            print(flat)
            generateCSV(flat, output)

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            generatePlot()

        print("all done!")

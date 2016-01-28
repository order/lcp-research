import config
import matplotlib.pyplot as plt

class TimeSeriesPlotter(config.Plotter):
    def display(self,data,save_file):
        print data.shape
        plt.plot(data)
        plt.show()
        

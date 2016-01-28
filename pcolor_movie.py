import config
import matplotlib.pyplot as plt

class TimeSeriesPlotter(config.Plotter):
    def display(self,data,save_file):
        plt.plot(data)
        plt.show()
        

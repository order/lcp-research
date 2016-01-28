import config
import utils.plotting
import matplotlib.pyplot as plt

class PcolorMoviePlot(config.Plotter):
    def display(self,data,save_file):

        # Data should be in (I, x, y)
        assert(3 == len(data.shape))

        utils.plotting.animate_frames(data,
                                      save_file=save_file)

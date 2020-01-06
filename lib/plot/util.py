
def colours():
    """ Returns the tableau20 colors as a list of three-dimensional tuples of
        numbers between 0 and 1

        Returns:
            List: list of tuples of color values given by rgb values between 0
                  and 1
    """

    # Tablea20 Colors for prettier plots
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # sclaed between 0 and 1 for matplotlib
    tableau20 = [(r/255., g/255., b/255.) for r,g,b in tableau20]

    return tableau20


def clean_plot(ax):
    """ Removes any unnecessary clutter in the plot figure, such as the frame
        borders around the plot
    """
    
    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

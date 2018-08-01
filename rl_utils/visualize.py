import numpy as np
from visdom import Visdom

class Visualizer(object):
    def __init__(self):
        self.viz = Visdom()
        self.global_lower_std, self.global_upper_std, self.global_min, self.global_max = 0, 0, 0, 0

    #return a line with the given points, color, and fill characteristics
    def get_line(self, x, y, name, color='transparent', isFilled=False, fillcolor='transparent'):
        if isFilled:
            fill = 'tonexty'
        else:
            fill = 'none'

        return dict(
            x=x,
            y=y,
            mode='lines',
            type='custom',
            line=dict(color=color),
            fill=fill,
            fillcolor=fillcolor,
            name=name,
            showlegend=False
        )

    #return a set of dots at the given points
    #they have a cool color scheme so that's fun
    def get_dots(self, x, y, name):
        return dict(
            x=x,
            y=y,
            mode='markers',
            type='custom',
            marker=dict(
                size = 6,
                color = x,
                colorscale = 'Viridis',
            ),
            name=name,
            showlegend=False
        )

    #create a plot of the loss
    def update_viz_loss(self, x, loss, color, env_name, win_name):
        loss = self.get_line(x, loss, 'loss', color=color)
        data = [loss]

        layout = dict(
            title=env_name + " Loss"
        )

        self.viz._send({'data': data, 'layout': layout, 'win': win_name + "-loss"})

    #create a plot with a line representing the mean and a shaded area around the mean representing the standard deviation of the rewards from all the actors
    def update_viz_mean(self, x, mean, std, colors, env_name, win_name):
        n = len(mean)
        upper_std = [0] * n
        lower_std = [0] * n
        for i in range(n):
            upper_std[i] = mean[i] + std[i]
            lower_std[i] = mean[i] - std[i]

        self.global_lower_std = min(lower_std)
        self.global_upper_std = max(upper_std)

        upper_line = self.get_line(x, upper_std, 'upper std', isFilled=True, fillcolor=colors[0])
        mean = self.get_line(x, mean, 'mean', color=colors[1])
        lower_line = self.get_line(x, lower_std, 'lower std')

        data = [lower_line, upper_line, mean]

        layout = dict(
            title=env_name + " Mean",
            yaxis=dict(range=[min(self.global_lower_std, self.global_min_r), max(self.global_upper_std, self.global_max_r)])
        )

        self.viz._send({'data': data, 'layout': layout, 'win': win_name + "-mean"})

    #create a plot with a line representing the median and two shaded regions representing the quartiles and most extreme values of the rewards from all the actors
    def update_viz_median(self, x, median, first_quartile, third_quartile, min_r, max_r, colors, env_name, win_name):
        self.global_min_r = min(min_r)
        self.global_max_r = max(max_r)

        max_line = self.get_line(x, max_r, 'max', isFilled=True, fillcolor=colors[0])
        upper_std = self.get_line(x, third_quartile, 'third quartile', isFilled=True, fillcolor=colors[1])
        median = self.get_line(x, median, 'median', color=colors[2])
        lower_std = self.get_line(x, first_quartile, 'first quartile')
        min_line = self.get_line(x, min_r, 'min')

        #'min_line' comes before 'max' so that 'max' can access 'min_line''s y values and stop the shading at them
        #same logic applies to 'lower_std' and 'upper_std', and they're placed after 'min' and 'max' so 'max' doesnt stop at one of them
        #'median' comes last so it doesnt stop the shading
        data = [min_line, max_line, lower_std, upper_std, median]

        layout = dict(
            title=env_name + " Median",
            yaxis=dict(range=[min(self.global_lower_std, self.global_min_r), max(self.global_upper_std, self.global_max_r)])
        )

        self.viz._send({'data': data, 'layout': layout, 'win': win_name + "-median"})

    #create a plot with dots at the given points
    #even though the line graphs are much easier to follow, this looks a lot cooler cause color scales and all that
    def update_viz_dots(x, y, data_type, env_name, win_name):

        dots = self.get_dots(x, y, data_type)
        data = [dots]

        layout = dict(
            title=env_name + " " + data_type
        )

        self.viz._send({'data': data, 'layout': layout, 'win': win_name + "-" + data_type})

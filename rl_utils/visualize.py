import numpy as np
from visdom import Visdom

viz = Visdom()

#this doesnt work cause global variables in python are weird
g_lower_std, g_upper_std, g_min_r, g_max_r = 0, 0, 0, 0

def get_line(x, y, name, color='transparent', isFilled=False, fillcolor='transparent'):
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

def get_dots(x, y, name):
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

def update_viz_mean(x, mean, std, colors, env_name, win_name):
    n = len(mean)
    upper_std = [0] * n
    lower_std = [0] * n
    for i in range(n):
        upper_std[i] = mean[i] + std[i]
        lower_std[i] = mean[i] - std[i]

    g_lower_std = min(lower_std)
    g_upper_std = max(upper_std)

    upper_line = get_line(x, upper_std, 'upper std', isFilled=True, fillcolor=colors[0])
    mean = median = get_line(x, mean, 'mean', color=colors[1])
    lower_line = get_line(x, lower_std, 'lower std')

    data = [lower_line, upper_line, mean]

    layout = dict(
        title=env_name + " Mean",
        yaxis=dict(range=[min(g_lower_std, g_min_r), max(g_upper_std, g_max_r)])
    )

    viz._send({'data': data, 'layout': layout, 'win': win_name + "-mean"})

def update_viz_median(x, median, first_quartile, third_quartile, min_r, max_r, colors, env_name, win_name):
    g_min_r = min(min_r)
    g_max_r = max(max_r)

    max_line = get_line(x, max_r, 'max', isFilled=True, fillcolor=colors[0])
    upper_std = get_line(x, third_quartile, 'third quartile', isFilled=True, fillcolor=colors[1])
    median = get_line(x, median, 'median', color=colors[2])
    lower_std = get_line(x, first_quartile, 'first quartile')
    min_line = get_line(x, min_r, 'min')

    #'min_line' comes before 'max' so that 'max' can access 'min_line''s y values and stop the shading at them
    #same logic applies to 'lower_std' and 'upper_std', and they're placed after 'min' and 'max' so 'max' doesnt stop at one of them
    #'median' comes last so it doesnt stop the shading
    data = [min_line, max_line, lower_std, upper_std, median]

    layout = dict(
        title=env_name + " Median",
        yaxis=dict(range=[min(g_lower_std, g_min_r), max(g_upper_std, g_max_r)])
    )

    viz._send({'data': data, 'layout': layout, 'win': win_name + "-median"})

def update_viz_dots(x, y, data_type, env_name, win_name):

    dots = get_dots(x, y, data_type)
    data = [dots]

    layout = dict(
        title=env_name + " " + data_type
    )

    viz._send({'data': data, 'layout': layout, 'win': win_name + "-" + data_type})

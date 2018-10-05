import plotly.offline as py
import numpy as np
import gpr

gp = gpr.GP()
data =  np.array([
                    [-2,-.5,.8,2.3,4,5.5],
                    [3,-2,8,2,1,5]
                ])

gp.fit(data[0],data[1])

layout_shape = ( (min(data[0])-1,max(data[0])+1), (min(data[1])-1,max(data[1])+1) )
grid_x = np.linspace(layout_shape[0][0], layout_shape[0][1],num=100)

post_m, post_s = gp.predict(grid_x)
post_std = np.sqrt( np.diagonal(post_s) )

#post_sample = np.random.multivariate_normal(post_m, post_s, size=30)

traces = [dict(
        showlegend = False,
        hoverinfo = 'none',
        visible = False,
        line=dict(color='rgb(255,20,50)', width=3),
        name = '𝜈 = '+str(step),
        x = grid_x,
        y = gp.update(step,1,1)) for step in np.arange(0,5,0.5)]
traces[1]['visible'] = True


steps = []
for i in range(len(traces)):
    step = dict(
        method = 'restyle',  
        args = ['visible', [False] * len(traces)],
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active = 10,
    currentvalue = {"prefix": "Frequency: "},
    pad = {"t": 50},
    steps = steps
)]

layout = dict(
        sliders = sliders,
        xaxis = dict( range=layout_shape[0] ),
        yaxis = dict( range=layout_shape[1] )
        )

fig = dict(data=traces, layout=layout)

py.plot(fig, filename='gaussian_processes_regression.html')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import gpr

data =  np.array([
                    [-2,-.5,.8,2,2.3,4,5.5],
                    [3,-2,8,2.1,2,1,5]
                ])


gp = gpr.GP()
gp.fit(data[0], data[1])

gp_axis = [
        min(data[0]) - 1,
        max(data[0]) + 1, 
        min(data[1]) - 2,
        max(data[1]) + 2 ]

n_grid_x = 201
n_samples = 15
grid_x = np.linspace(gp_axis[0], gp_axis[1], num=n_grid_x)
dummy_init = grid_x*0

post_m, post_s = gp.predict(grid_x)
post_std = np.sqrt( np.diagonal(post_s) )

main_axis = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)

curve_area = plt.fill_between(grid_x,dummy_init,dummy_init)

plt.scatter(data[0], data[1], c='b')
curve, = plt.plot(grid_x, dummy_init, c='r', lw=1, label='mean')

curve_samples = [plt.plot(grid_x, dummy_init, lw=1, alpha=.4, c='k')[0] \
                for i in range(n_samples)]

plt.axis(gp_axis)
from matplotlib.patches import Patch
curve_legend = Patch(color='b', alpha=.4, label='std')
plt.legend(handles=[curve,curve_legend], loc=1)

rbf_axis = plt.subplot2grid((3, 3), (2, 2))
rbf_axis.set_title('RBF kernel', fontsize=8)
rbf_plot_x = np.linspace(-1.2,1.2,20)**3
rbf_plot, = plt.plot(rbf_plot_x, rbf_plot_x*0 )
plt.xlabel(r'$s_0*\exp(-\frac{s_1}{2}|x_i-x_j|^{s_2})$')

plt.tight_layout()
fig = plt.gcf()


def toggle_vis(state):

    update(0)

    if state == 'sample':
        main_axis.get_legend().set_visible(False)
        curve.set_linestyle('')
        curve_area.set_facecolor('none')
        for c in curve_samples:
            c.set_linestyle('-')
    else:
        main_axis.get_legend().set_visible(True)
        curve.set_linestyle('-')
        curve_area.set_facecolor('b')
        for c in curve_samples:
            c.set_linestyle('')

def update(val):

    bi = np.round(np.exp(slider[0].val), 3)
    s0 = np.round(np.exp(slider[1].val), 2)
    s1 = np.round(np.exp(slider[2].val), 2)
    s2 = np.round(slider[3].val, 1)
    slider[0].valtext.set_text(bi)
    slider[1].valtext.set_text(s0)
    slider[2].valtext.set_text(s1)
    slider[3].valtext.set_text(s2)

    post_m, post_s = gp.update(bi,s0,s1,s2)
    post_std = np.sqrt( np.diagonal(post_s) )

    if button.value_selected == 'sample':
        for c in curve_samples:
            c.set_ydata(np.random.multivariate_normal(post_m, post_s))
    else:
        curve.set_ydata(post_m)

        global curve_area
        curve_area.remove()
        curve_area = main_axis.fill_between(grid_x, post_m+post_std, 
                                post_m-post_std, color='b', alpha=.3, label='std')
        curve_area.set_edgecolor('none')



    rbf_y = gp.kernel_matrix([0],rbf_plot_x)
    rbf_plot.set_ydata(rbf_y)
    rbf_axis.set_ylim(0, max(rbf_y)+1)

    plt.draw()

    fig.canvas.draw_idle()


slider_ax = [
        plt.axes([.26, .31, .36, .03]),
        plt.axes([.26, .24, .36, .03]),
        plt.axes([.26, .19, .36, .03]),
        plt.axes([.26, .14, .36, .03]) ]
slider = [
        Slider(slider_ax[0], r'precision: $\beta^{-1}$', -7.7, 1.38, valinit=-7.7, valstep=.1),
        Slider(slider_ax[1], r'$s_0$', -2.3, 2.31, valinit=0, valstep=.1),
        Slider(slider_ax[2], r'$s_1$', -1.6, 2.8, valinit=0, valstep=.1),
        Slider(slider_ax[3], r'$s_2$', 0.5, 2, valinit=2, valstep=.1) ]

button_ax = plt.axes([.07, .14, .12, .13])
button = RadioButtons(button_ax, ('mean','sample'), active=0)
button.on_clicked(toggle_vis)

for s in slider:
    s.on_changed(update)


toggle_vis('mean')
plt.show()


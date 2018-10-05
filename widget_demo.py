import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import gpr


### toy data
toy_data_1 = np.array([ [-2, -.5, .8, 2, 2.3, 4, 5.5],
                    [1.5, -1, 4, 1.1, 1, .5, 2.5] ])

toy_func = np.sinc
sample_x = np.random.uniform(-2, 8, 20)
noise_y = np.random.normal( 0, .15, [len(sample_x)] ).T
toy_data_2 = np.array([ sample_x, toy_func(sample_x) + noise_y ]) 

data = toy_data_1

### fit data
gp = gpr.GP()
gp.fit(data[0], data[1])


### widget start
n_grid_x = 201
n_samples = 15

### build window and graphs
data_view = [
        min(data[0]) - 1, max(data[0]) + 1, 
        min(data[1]) - 1, max(data[1]) + 1 ]

#fig = plt.figure()
main_axis = plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=2)

text_mse = main_axis.text(.85, .05, '', 
                verticalalignment='bottom', transform=main_axis.transAxes)

plt.axis(data_view)

### create placeholders for later plot
plt.scatter(data[0], data[1], c='b')    # main data plot

grid_x = np.linspace(data_view[0], data_view[1], num=n_grid_x)
dummy_init = grid_x*0

#XXX
#plt.plot(grid_x, toy_func(grid_x))

curve, = plt.plot(grid_x, dummy_init, c='r', lw=1, label='mean')
curve_area = plt.fill_between(grid_x, dummy_init, dummy_init)

curve_samples = [plt.plot(grid_x, dummy_init, lw=1, alpha=.4, c='k')[0] \
                for i in range(n_samples)]

### set legend
from matplotlib.patches import Patch
curve_legend = Patch(color='b', alpha=.4, label='std')
plt.legend(handles=[curve,curve_legend], loc=1)

### create kernel window
rbf_axis = plt.subplot2grid((3,3), (2,2))
rbf_axis.set_title('RBF kernel', fontsize=8)
rbf_plot_x = np.linspace(-1.3, 1.3, 25)**3      # finer plot around 0
rbf_plot, = plt.plot(rbf_plot_x, rbf_plot_x*0 )
plt.xlabel(r'$s_0*\exp(-\frac{s_1}{2}|x_i-x_j|^{s_2})$')

plt.tight_layout()


# main update for graph, redraw
def update(val):

    # get slider values
    bi = np.round(np.exp(slider[0].val), 3)
    s0 = np.round(np.exp(slider[1].val), 2)
    s1 = np.round(np.exp(slider[2].val), 2)
    s2 = np.round(slider[3].val, 1)
    slider[0].valtext.set_text(bi)
    slider[1].valtext.set_text(s0)
    slider[2].valtext.set_text(s1)
    slider[3].valtext.set_text(s2)

    # calculate new regression with updated kernel
    post_m, post_s = gp.update(grid_x, bi, s0, s1, s2)
    post_std = np.sqrt( np.diagonal(post_s) )

    # main view
    if view_button.value_selected == 'mean':

        curve.set_ydata(post_m)     # update mean curve
        text_mse.set_text('rmse: ' + str(np.round(gp.rmse(), 2)))

        # curve_area is deleted and recreated, since moving all
        # poly-shapes is impractical
        global curve_area
        curve_area.remove() 
        curve_area = main_axis.fill_between(
                                grid_x, post_m+post_std, post_m-post_std, 
                                color='b', alpha=.3, label='std')
        curve_area.set_edgecolor('none')

    else:       # sample from calculated distribution
        for c in curve_samples:
            c.set_ydata(np.random.multivariate_normal(post_m, post_s))

    # adjust kernel graph
    rbf_y = gp.kernel_matrix([0], rbf_plot_x)
    rbf_plot.set_ydata(rbf_y)
    rbf_axis.set_ylim(0, max(rbf_y)+1)

    plt.draw()


### create slider widgets
slider_ax = [
        plt.axes([.26, .31, .36, .03]),
        plt.axes([.26, .24, .36, .03]),
        plt.axes([.26, .19, .36, .03]),
        plt.axes([.26, .14, .36, .03]) ]
slider = [
        Slider(slider_ax[0], r'precision: $\beta^{-1}$', 
                                    -7.7, 1.38, valinit=-7, valstep=.1),
        Slider(slider_ax[1], r'$s_0$', -2.3, 2.31, valinit=0, valstep=.1),
        Slider(slider_ax[2], r'$s_1$', -1.6, 2.8, valinit=0, valstep=.1),
        Slider(slider_ax[3], r'$s_2$', 0.5, 2, valinit=2, valstep=.1) ]

for s in slider:
    s.on_changed(update)

### create radio button
button_ax = plt.axes([.07, .14, .12, .13])
view_button = RadioButtons(button_ax, ('mean','sample'), active=0)

# is called when radiobutton is clicked
def toggle_vis(state):  

    # update graph
    update(0)

    # toggle view and legend
    if state == 'sample':
        main_axis.get_legend().set_visible(False)
        text_mse.set_visible(False)
        curve.set_linestyle('')
        curve_area.set_facecolor('none')
        for c in curve_samples:
            c.set_linestyle('-')
    else:
        main_axis.get_legend().set_visible(True)
        text_mse.set_visible(True)
        curve.set_linestyle('-')
        curve_area.set_facecolor('b')
        for c in curve_samples:
            c.set_linestyle('')


view_button.on_clicked(toggle_vis)

toggle_vis('mean')
plt.show()


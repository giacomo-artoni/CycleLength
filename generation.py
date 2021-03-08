import names
import random
import sys
import numpy as np
import pandas as pd

from scipy.stats import gamma, multivariate_normal, norm, randint, t

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils.plotting as plotting
import utils.math_ops as math_ops


class Generator:

    def __init__(self, gen='multi_gauss'):
        self.cl_clv_distr, self.cl_distr, self.clv_distr = None, None, None
        if gen == 'multi_gauss':
            means, sigmas = [30, 3], [[5., 1.5], [1.5, 2.]]
            self.cl_clv_distr = multivariate_normal(means, sigmas)
        elif gen == 'double_gamma':
            self.cl_distr = gamma(30)
            self.clv_distr = gamma(3)
        elif gen == 'gamma_gauss':
            self.cl_distr = gamma(30)
            self.clv_distr = norm(3, 0.5)
        self.ns = randint(5, 50)
        self.individual_cl_distr = norm

    def get_pdf(self, v):
        if self.cl_clv_distr:
            return self.cl_clv_distr.pdf(v)
        elif self.cl_distr and self.clv_distr:
            return np.array([[self.cl_distr.pdf(x) * self.clv_distr.pdf(y) for x, y in sub_v] for sub_v in v])

    def generate_people(self, n=10):
        people_coords = self.get_random_variates(n)
        people_names = [names.get_full_name(gender='female') for _ in range(n)]
        colors = [plt.cm.Spectral(index) for index in np.linspace(0, 1, n)]
        infos = []
        for (mu, sigma), name, color in zip(people_coords, people_names, colors):
            entries = self.ns.rvs()
            vals = self.individual_cl_distr(mu, sigma).rvs(entries)
            infos.append(dict(name=name, color=color, true_cl=mu, true_clv=sigma, cls=list(vals)))
        return pd.DataFrame.from_dict(infos)

    def get_random_variates(self, n):
        if self.cl_clv_distr:
            res = self.cl_clv_distr.rvs(n)
        elif self.cl_distr and self.clv_distr:
            res = [(x, y) for x, y in zip(self.cl_distr.rvs(n), self.clv_distr.rvs(n))]
        else:
            return []
        if n == 1:
            res = [res]
        neg_sigmas = sum([y < 0 for x, y in res])
        if neg_sigmas > 0:
            pos_res = res[res[:, 1] > 0]
            return np.concatenate((pos_res, self.get_random_variates(neg_sigmas)), axis=0)
        else:
            return res


def show_generation_result():

    # Define 2d model for cycle length and cycle length variability
    random.seed(19101985)
    np.random.seed(28111987)
    plt.style.use('seaborn-talk')
    fig_x, fig_y = 6, 6
    x, y = np.mgrid[20:40:0.1, 0:7:0.05]
    par_space_points = np.dstack((x, y))
    toy_generator = Generator(gen='double_gamma')
    x_range, y_range = (25, 40), (1, 5)
    z = toy_generator.get_pdf(par_space_points)

    # Generate a few points in this parameter space
    data = toy_generator.generate_people(10)

    # Plotting
    with PdfPages('plots/multivariate_test.pdf') as pages:

        # 2D model
        fig = plt.figure(1, figsize=(fig_x, fig_y))
        ax = fig.add_subplot(111)
        ax.contour(x, y, z, 25, linewidths=0.7, cmap=plt.cm.Greys, linestyles='dashed', zorder=0)
        for x, y, name, color in zip(data['true_cl'], data['true_clv'], data['name'], data['color']):
            plotting.add_point(ax=ax, pos=(x, y), x_range=x_range, y_range=y_range, name=name, color=color)
        plotting.set_axis_info(ax, x_range=x_range, y_range=y_range,
                               title_x='Cycle Length', title_y='Cycle Length Variability')
        pages.savefig(fig, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Individual gaussians
        for index, row in data.iterrows():
            fig = plt.figure(1, figsize=(fig_x, fig_y))
            ax = fig.add_subplot(111)
            vals, color = row['cls'], row['color']
            kwargs = dict(color=color, linewidth=1.5)
            counts, bins, patches = ax.hist(vals, density=True, histtype='stepfilled', bins=40, alpha=0.3, **kwargs)
            bin_cs = np.array(math_ops.get_bin_centers(bins))
            authors, labels = [patches[0]], ['Generated values ({})'.format(len(vals))]
            try:
                bf_pars = t.fit(vals)
                best_fit, = ax.plot(bin_cs, t.pdf(bin_cs, *bf_pars), linestyle='--', **kwargs)
                nothing = plotting.get_dummy_element()
                authors += [nothing, best_fit, nothing]
                labels += ['', 'Best fit', '$\mu\' = {1:.2f}, \sigma\' = {2:.2f}, \\nu\' = {0:.1f}$'.format(*bf_pars)]
                true_pars = (row['true_cl'], row['true_clv'])
                true_vals, = ax.plot(bin_cs, norm.pdf(bin_cs, *true_pars), linestyle='-', **kwargs)
                authors += [nothing, true_vals, nothing]
                labels += ['', 'True distribution', '$\mu = {0:.2f}, \sigma = {1:.2f}$'.format(*true_pars)]
            except RuntimeError:
                print('Fit failed ---> ', len(vals))
            plotting.set_axis_info(ax, title_x='Cycle Length', title_y='Arbitrary Units')
            ax.legend(authors, labels, loc='upper left', fontsize=10, framealpha=0.1)
            pages.savefig(fig, bbox_inches='tight', pad_inches=0.1)
            plt.close()


if __name__ == '__main__':
    sys.exit(show_generation_result())

import numpy as np
import pandas as pd
from copy import copy, deepcopy
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import lib.python_model_renderer.render_galaxy as rg
from progress.spinner import Spinner


def get_params(n_arms):
    sersic_keys = ('i0', 'rEff', 'axRatio', 'n', 'c')
    spiral_keys = ('i0', 'spread', 'falloff')
    params = {
        'disk': sersic_keys[:2],
        'bulge': sersic_keys[:3],
        'bar': sersic_keys[:],
    }
    params.update({
        'spiral.{}'.format(i): spiral_keys
        for i in range(n_arms)
    })
    return params


FIT_PARAMS = {
    'disk':   ('i0', 'rEff', 'axRatio'),
    'bulge':  ('i0', 'rEff', 'axRatio', 'n'),
    'bar':    ('i0', 'rEff', 'axRatio', 'n', 'c'),
    'spiral': ('i0', 'spread', 'falloff'),
}

class Model():
    def __init__(self, model, galaxy_data, psf=None, pixel_mask=None):
        self.base_model = deepcopy(model)
        self.model = deepcopy(model)
        self.data = galaxy_data
        self.psf = psf if psf is not None else np.ones((1, 1))
        self.pixel_mask = (
            pixel_mask if pixel_mask is not None
            else np.ones_like(galaxy_data)
        )
        self.n_arms = len(model['spiral'])
        self.params = get_params(self.n_arms)
        self.comps = self.calculate_components(model)
        self.p, self.__p_key = self.construct_p(model)
        self.change_map = np.array([i[0] for i in self.__p_key])

    def construct_p(self, model):
        p = np.array([
            model[comp][k]
            for comp in ('disk', 'bulge', 'bar')
            for k in FIT_PARAMS[comp]
            if model[comp] is not None
        ] + [
            model['spiral'][i][1][k]
            for i in range(len(model['spiral']))
            for k in FIT_PARAMS['spiral']
        ])
        p_key = [
            (comp, k)
            for comp in ('disk', 'bulge', 'bar')
            for k in FIT_PARAMS[comp]
            if model[comp] is not None
        ] + [
            ('spiral', i, k)
            for i in range(len(model['spiral']))
            for k in FIT_PARAMS['spiral']
        ]
        return p, p_key

    def calculate_components(self, model, oversample_n=5):
        disk_arr = self.render_component('disk', model, oversample_n)
        bulge_arr = self.render_component('bulge', model, oversample_n)
        bar_arr = self.render_component('bar', model, oversample_n)
        spiral_arr = self.render_component('spiral', model, oversample_n)
        return np.stack((disk_arr, bulge_arr, bar_arr, spiral_arr))

    def render_component(self, comp_name, model, oversample_n=5):
        if comp_name == 'spiral':
            if len(model['spiral']) == 0:
                return np.zeros_like(self.data)
            return np.add.reduce([
                rg.spiral_arm(
                    *s,
                    model['disk'],
                    image_size=self.data.shape[0],
                )
                for s in model['spiral']
            ])
        return rg.sersic_comp(
            model[comp_name],
            image_size=self.data.shape[0],
            oversample_n=oversample_n,
        )

    def detect_changes(self, p, old_p):
        mask = abs(p - old_p) > 0
        return np.unique(self.change_map[mask])

    def update_model(self, old_model, p):
        new_model = deepcopy(old_model)
        for i, v in enumerate(self.__p_key):
            try:
                new_model[v[0]][v[1]][1][v[2]] = p[i]
            except (TypeError, IndexError):
                new_model[v[0]][v[1]] = p[i]
        return rg.sanitize_model(new_model)

    def render(self, model):
        return np.add.reduce(
            self.calculate_components(model)
        )

    def render_from_p(self, new_p):
        to_update = self.detect_changes(new_p, self.p)
        new_model = self.update_model(self.base_model, new_p)
        for i, c in enumerate(('disk', 'bulge', 'bar', 'spiral')):
            if c in to_update:
                self.comps[i] = self.render_component(c, new_model)
        self.p = new_p
        self.model = new_model
        return np.add.reduce(self.comps)

    def to_df(self, model=None):
        m = model if model is not None else self.base_model
        idx = ('disk', 'bulge', 'bar')

        c = pd.DataFrame([m[i] if m[i] is not None else {} for i in idx], index=idx)
        spirals = pd.DataFrame([i[1] for i in m['spiral']])
        spirals.index.name = 'Spiral number'
        return c, spirals

    def _repr_html_(self, model=None):
        c, spirals = self.to_df(model)
        return c.to_html() + spirals.to_html()


class ModelFitter():
    def __init__(self, base_model, galaxy_data, psf=None, pixel_mask=None):
        self.model = Model(base_model, galaxy_data, psf, pixel_mask)

    def loss(self, rendered_model):
        pixel_mask = self.model.pixel_mask
        Y = rg.convolve2d(
            rendered_model, self.model.psf, mode='same', boundary='symm'
        ) * pixel_mask
        return mean_squared_error(Y.flatten(), 0.8 * (self.model.data * pixel_mask).flatten())

    def fit(self, oversample_n=5, *args, **kwargs):
        md = deepcopy(self.model.base_model)
        p0, p_key = self.model.construct_p(md)
        # p0 = np.array([
        #     md[comp][k]
        #     for comp in ('disk', 'bulge', 'bar')
        #     for k in FIT_PARAMS[comp]
        # ] + [
        #     md['spiral'][i][1][k]
        #     for i in range(self.model.n_arms)
        #     for k in FIT_PARAMS['spiral']
        # ])

        def _f(p):
            return self.loss(self.model.render_from_p(p))

        # return _f, p0
        res = minimize(_f, p0, *args, **kwargs)
        return self.model.update_model(
            md, res['x']
        ), res

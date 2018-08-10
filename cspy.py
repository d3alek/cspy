import random
from scipy import optimize, signal
import numpy as np
from lmfit import models
import scipy.io
import matplotlib.pyplot as plt

def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

def update_spec_from_peaks(spec, model_indicies, peak_widths=(10, 25), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies

def print_best_values(spec, output):
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma']
    }
    best_values = output.best_values
    print('center    model   amplitude     sigma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')

def fit(mat_file, gaussians, peak_widths):
    mat = scipy.io.loadmat(mat_file)
    tables = list(mat.keys())
    print('\n'.join(map(lambda t: '%d: %s' % (t[0],t[1]), enumerate(tables))))
    table_index = int(input("Write a number to select the table to model: "))
    table = tables[table_index]
    print('Fitting %s with %d gaussians with peak widths %d' % (table, gaussians, peak_widths))

    spec = {
        'x': mat[table][:,0],
        'y': mat[table][:,1],
        'model': [ {'type': 'GaussianModel'} for i in range(gaussians)]
    }

    peaks_found = update_spec_from_peaks(spec, range(len(spec['model'])), peak_widths=(peak_widths,))
    fig, ax = plt.subplots()
    ax.scatter(spec['x'], spec['y'], s=4)
    for i in peaks_found:
        ax.axvline(x=spec['x'][i], c='black', linestyle='dotted')
    model, params = generate_model(spec)

    output = model.fit(spec['y'], params, x=spec['x'])
    fig, gridspec = output.plot(data_kws={'markersize':  1})

    fig, ax = plt.subplots()
    ax.scatter(spec['x'], spec['y'], s=4)
    components = output.eval_components(x=spec['x'])
    for i, model in enumerate(spec['model']):
        ax.plot(spec['x'], components[f'm{i}_'])


    print_best_values(spec, output)

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mat_file')
    parser.add_argument('-g', '--gaussians', default=2, type=int)
    parser.add_argument('-w', '--peak-widths', default=50, type=int)
    args = parser.parse_args()

    fit(args.mat_file, args.gaussians, args.peak_widths)


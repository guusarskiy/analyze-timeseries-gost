import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scipy.stats import levene
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from diptest import diptest
from fitter import Fitter
import sys
import traceback
import json

ac_distributions=['norm', 'lognorm', 'rayleigh', 'foldnorm', 'weibull_min', 'weibull_max', 'pearson3']
b_distributions=['lognorm', 'rayleigh', 'foldnorm', 'weibull_min', 'weibull_max', 'pearson3']

def analyze_position_change(time_series, num_segments=10, alpha=0.05):
    ts = pd.Series(time_series)
    n = len(ts)
    segment_size = n // num_segments
    segments = []

    for i in range(num_segments):
        segment = ts[i*segment_size:(i+1)*segment_size]
        segments.append(segment.values)

    stat, p_value = kruskal(*segments)

    if p_value > alpha:
        position_class = 'const'
    else:
        position_class = 'change'

    return position_class

def analyze_variability_change(time_series, num_segments=10, alpha=0.05):
    ts = pd.Series(time_series)
    n = len(ts)
    segment_size = n // num_segments
    segments = []

    for i in range(num_segments):
        segment = ts[i*segment_size:(i+1)*segment_size]
        segments.append(segment.values)

    stat, p_value = levene(*segments)

    if p_value > alpha:
        variability_class = 'const'
    else:
        variability_class = 'change'

    return variability_class

def classify_abcd(position_class, variability_class):
    if position_class == 'const' and variability_class == 'const':
        model_class = 'A'
    elif position_class == 'const' and variability_class == 'change':
        model_class = 'B'
    elif position_class == 'change' and variability_class == 'const':
        model_class = 'C'
    elif position_class == 'change' and variability_class == 'change':
        model_class = 'D'
    else:
        model_class = 'unknown'
    return model_class

def classify_time_series_abcd(time_series, num_segments=4, alpha=0.05):
    position_class = analyze_position_change(time_series, num_segments, alpha)
    variability_class = analyze_variability_change(time_series, num_segments, alpha)
    model_class = classify_abcd(position_class, variability_class)

    return model_class
    
def test_unimodality(data, alpha=0.05):
    dip_statistic, p_value = diptest(data)
    if p_value > alpha:
        return True  
    else:
        return False
    
def get_top_fitter(data, distributions):
    fitter_model = Fitter(data, distributions=distributions, timeout=30)
    fitter_model.fit()
    top1 = fitter_model.get_best()
    top1_dist = list(top1.keys())[0]

    return top1_dist

def get_distr_class(fitter_class):
    if fitter_class == 'norm':
        return 'Normal distribution'
    elif fitter_class == 'lognorm':
        return 'Log-normal distribution'
    elif fitter_class == 'rayleigh':
        return 'Rayleigh distribution'
    elif fitter_class == 'foldnorm':
        return 'Folded normal distribution'
    elif fitter_class == 'weibull_min':
        return 'Weibull distribution (right-tailed)'
    elif fitter_class == 'weibull_max':
        return 'Weibull distribution (left-tailed)'
    elif fitter_class == 'pearson3':
        return 'Pearson distribution system'
    elif fitter_class == 'multimodal':
        return 'Mixed distribution'
    else:
        return 'Unknown distribution'
    
def analyze_position_trend_regression(time_series, alpha=0.05):
    ts = pd.Series(time_series)
    n = len(ts)
    x = np.arange(n)
    x = sm.add_constant(x)
    
    model = sm.OLS(ts, x).fit()
    p_value = model.pvalues[1]
    
    dw_stat = durbin_watson(model.resid)

    autocorrelation = dw_stat < 1.5 or dw_stat > 2.5
    
    if p_value < alpha and not autocorrelation:
        position_change = 'system'
    else:
        position_change = 'random'
    
    return position_change

def classify_subclass(time_series, num_segments=4, alpha=0.05):
    ts = pd.Series(time_series)

    model_class = classify_time_series_abcd(ts, num_segments, alpha)

    if (model_class == 'A'):
        _distribution = get_top_fitter(ts, ac_distributions)

        if (_distribution == 'norm'):
            subclass = 'A1'
        else:
            subclass = 'A2'

    elif (model_class == 'B'):
        _distribution = get_top_fitter(ts, b_distributions)
        subclass = 'B'

    elif (model_class == 'C'):
        position_trend = analyze_position_trend_regression(ts)
        unimodeltest = test_unimodality(ts)
        _distribution = get_top_fitter(ts, ac_distributions)

        if (unimodeltest) and (position_trend == 'random'):
            if (_distribution == 'norm'):
                subclass = 'C1'
            else:
                subclass = 'C2'
        else:
            if (position_trend == 'system'):
                subclass = 'C3'
            else:
                subclass = 'C4'

    elif (model_class == 'D'):
        _distribution = 'multimodal'
        subclass = 'D'

    else:
        subclass = 'unknown'

    distr_class = get_distr_class(_distribution)

    return subclass, distr_class

def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_message = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
        print(error_message, file=sys.stderr)
        return None
    
def parse_json_to_numpy(json_string):
    try:
        data_list = json.loads(json_string)  
        np_array = np.array(data_list, dtype=float)  
        return np_array
    except Exception as e:
        print(f"Error parsing JSON: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    try:
        json_string = sys.argv[1]

        time_series = parse_json_to_numpy(json_string)
        if time_series is None:
            sys.exit(1)  

        result = safe_execute(classify_subclass, time_series)
        if result is None:
            sys.exit(1)  

        result_output = {
            "modelName": result[0],
            "distribName": result[1]
        }
        print(json.dumps(result_output, ensure_ascii=False))
    except Exception as e:
        error_message = f"Critical Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message, file=sys.stderr)
        sys.exit(1)

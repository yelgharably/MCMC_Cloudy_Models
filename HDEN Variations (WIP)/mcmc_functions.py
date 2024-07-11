import numpy as np
from scipy import interpolate
from k_lambda_py3 import ebv2ext

def cloudy_grid(line, U_iter, Zneb_iter, Zst_iter, LLT, U_values, Zneb_values, Zst_values):
    org_linelist = np.zeros((U_iter, Zneb_iter, Zst_iter))
    for lineListTable in LLT:
        org_linelist[:,:,:] = lineListTable[line].reshape(U_iter, Zneb_iter, Zst_iter)
    return org_linelist

def model_function(theta, lines, U_values, Zneb_values, Zst_values, LLT, lines_vac):
    U, Zneb, Zst, ebv = theta
    
    colnames = LLT[0].colnames
    normalized_colnames = np.array([name.strip().lower() for name in colnames])

    target_col = 'hi_6563'.strip().lower()

    if target_col in normalized_colnames:
        hi_6563_idx = np.where(normalized_colnames == target_col)[0][0]
    else:
        raise ValueError(f"'{target_col}' not found in column names: {normalized_colnames}")
    
    def interpolator(line):
        flux_grid = cloudy_grid(line, len(U_values), len(Zneb_values), len(Zst_values), LLT, U_values, Zneb_values, Zst_values)
        RGI = interpolate.RegularGridInterpolator((U_values, Zneb_values, Zst_values), flux_grid, method='linear', bounds_error=False)
        RGI_eval = RGI([U, Zneb, Zst])
        return RGI_eval.item()
    
    line_exts = ebv2ext(wave=lines_vac, ebv=ebv, dlaw='ccm')
    flux_value_H = interpolator('HI_6563') * line_exts[7]
    
    fluxes = []
    # EDIT THIS LATER TO len(lines) OR ELSE...
    for i in range(22):
        flux_values = interpolator(line=lines[i]) * (100 / flux_value_H) * line_exts[i]
        fluxes.append(flux_values)
    
    return fluxes

def lnlike(theta, x, y, yerr, U_values, Zneb_values, Zst_values, LLT, lines_vac):
    model_fluxes = model_function(theta, x, U_values, Zneb_values, Zst_values, LLT, lines_vac)
    return -0.5 * np.sum(((y - model_fluxes) / yerr) ** 2)

def lnprior(theta, U_min, U_max, Zneb_min, Zneb_max, Zst_min, Zst_max):
    U, Zneb, Zst, ebv = theta
    if U_min <= U <= U_max and Zneb_min <= Zneb <= Zneb_max and Zst_min <= Zst <= Zst_max and 0.0 <= ebv <= 0.47:
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, yerr, U_min, U_max, Zneb_min, Zneb_max, Zst_min, Zst_max, U_values, Zneb_values, Zst_values, LLT, lines_vac):
    lp = lnprior(theta, U_min, U_max, Zneb_min, Zneb_max, Zst_min, Zst_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, U_values, Zneb_values, Zst_values, LLT, lines_vac)

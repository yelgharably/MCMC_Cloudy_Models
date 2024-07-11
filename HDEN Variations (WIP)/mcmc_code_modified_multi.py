import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import emcee as mc
import corner
from makeCloudyTable import makeCloudyTable
import os
import tkinter as tk
from tkinter import filedialog
from scipy import interpolate
import time
from multiprocessing import Pool, cpu_count
from mcmc_functions import cloudy_grid, lnprob

def run_sampler(args):
    sampler, p0, niter = args
    sampler.run_mcmc(p0, niter)
    return sampler

def main(p0, nwalkers, niter, ndim, lnprob, data, pool):
    sampler = mc.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)

    print("Running burn-in...")
    burn_in_start = time.time()
    p0, _, _ = sampler.run_mcmc(p0, 300)
    burn_in_end = time.time()
    sampler.reset()
    print(f"Burn-in completed in {burn_in_end - burn_in_start:.2f} seconds")

    print("Running production...")
    production_start = time.time()
    sampler.run_mcmc(p0, niter)
    production_end = time.time()
    print(f"Production completed in {production_end - production_start:.2f} seconds")

    pos, prob, state = sampler.chain[:, -1, :], sampler.lnprobability[:, -1], sampler.random_state

    return sampler, pos, prob, state

def get_directory_path():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select Directory Containing 'sm' Files")
    return directory.replace("\\", "/")

def get_root_from_directory(directory):
    in_grd_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (file.startswith('big') and (file.endswith(".out") or file.endswith(".grd"))):
                in_grd_files.append(os.path.join(root, file).replace("\\", "/"))
    return np.sort(in_grd_files)

def get_unique_file_names(file_paths):
    file_names = set()  # avoiding repeats
    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_name.endswith("grid"):
            file_name = file_name[:-4]  # Remove "grid" from the end
        file_names.add(file_name)
    return np.sort(list(file_names))

def get_zst_from_root(file_roots):
    Zst = []
    for file in file_roots:
        Zst.append(float(file[-2:]) * (1e-3))
    return np.sort(Zst)

def get_eddrat_from_root(file_roots):
    SED = []
    for file in file_roots:
        if file.startswith('biggridHIGHEST'):
            SED.append(0.65)
        elif file.startswith('biggridHIGH'):
            SED.append(-0.03)
        elif file.startswith('biggridMID'):
            SED.append(-0.55)
        elif file.startswith('biggridLOW'):
            SED.append(-1.15)
        else:
            raise Exception(f"file {file} has wrong format.")
    return np.sort(SED)

def extract_lines(table, columns):
    return table[columns]

if __name__ == "__main__":
    start_time = time.time()

    directory = get_directory_path()
    files = np.sort(get_unique_file_names(get_root_from_directory(directory)))
    print(f"The files are: {files}, with length: {len(files)}")
    SED_values = get_eddrat_from_root(files)
    print(f"SED_values is: {SED_values}")
    files_path = [directory + '/' + file for file in files]

    LLT = []
    for i in range(len(files)):
        _, _ = makeCloudyTable(files_path[i], linelistsuffix='linelist.output', gridsuffix='grid.grd', write_table=False)
        lineListTable, _ = makeCloudyTable(files_path[i])
        LLT.append(lineListTable)
        print(f"LLT[{i}].colnames = {LLT[i].colnames}, The length of the columns is: {len(LLT[i].colnames)}")
        print(f"The file is: {files[i]}")
        print(f"LLT[{i}] = {LLT[i]}")
    if len(LLT) == 0:
        raise ValueError("LLT is empty.")

    # lines_to_extract = ['grid_iter', 'IONIZATIO', 'HDEN=%f','HI_1216','Blnd_1240','Blnd_1549','HeII_1640','HI_4861','OIII_4959','OIII_5007']
    # LLT = [extract_lines(table,lines_to_extract) for table in LLT]
    # print(f"Final LLT is: {LLT}")

    U_values = np.array(np.unique(LLT[0]['IONIZATIO']))
    hden_values = np.array(np.unique(LLT[0]['HDEN']))
    Zneb_values = np.array(np.unique(LLT[0]['METALS']))

    U_iter = len(np.unique(LLT[0]['IONIZATIO']))
    hden_iter = len(np.unique(LLT[0]['HDEN']))
    Zneb_iter = len(np.array(np.unique(LLT[0]['METALS'])))
    SED_iter = len(SED_values)
    print(f"U_values are: {U_values} with length: {U_iter}\nHDEN_values are: {hden_values}, with length: {hden_iter}\nSED_values are: {SED_values}, with length: {SED_iter}\nZneb_iter is: {Zneb_values} with length {Zneb_iter}")
 
    U_min = np.min(U_values)
    U_max = np.max(U_values)
    hden_min = np.min(hden_values)
    hden_max = np.max(hden_values)
    Zneb_min = np.min(Zneb_values)
    Zneb_max = np.max(Zneb_values)
    SED_min = np.min(SED_values)
    SED_max = np.max(SED_values)

    data = np.genfromtxt(r"D:\Undergraduate Life\Summer 2024\Trainor_Research\JHslit_lines.csv", delimiter=',', skip_header=1)
    line_names = data[:,0]
    rfluxes_measured = data[:, 1]
    rfluxes_err = data[:,2]
    lines_vac = data[:, 3]

    CC_err = rfluxes_err
    CC_data = [LLT[0].colnames[4:], rfluxes_measured, CC_err]

    cloudy_grids = {line: cloudy_grid(line, U_iter, hden_iter, Zneb_iter, LLT, U_values, hden_values, Zneb_values) for line in CC_data[0]}
    interpolators = {line: interpolate.RegularGridInterpolator((U_values, hden_values, Zneb_values), cloudy_grids[line], method='linear', bounds_error=False) for line in CC_data[0]}

    line = CC_data[0]
    flux = CC_data[1]
    data = (line, flux, CC_err, U_min, U_max, hden_min, hden_max, Zneb_min, Zneb_max, U_values, hden_values, Zneb_values, LLT, lines_vac)

    nwalkers = 100 
    niter = 1000
    initial = np.array([-2.5, 0.5, 0.010, 0.25])
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

    with Pool(processes=cpu_count()) as pool:
        sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data, pool)

    samples = sampler.flatchain
    Us, hdens, Znebs, ebvs = sampler.flatchain.T
    mcmc_parameters = np.array([Us, hdens, Znebs, ebvs])
    np.save(file='mcmc_output', arr=mcmc_parameters)

    U_mean, hden_mean, Zneb_mean, ebv_mean = np.mean(Us), np.mean(hdens), np.mean(Znebs), np.mean(ebvs)

    def plot_corner(samples):
        corner.corner(samples, labels=['U', 'HDEN', 'SED', 'ebv'], truths=[U_mean, hden_mean, Zneb_mean, ebv_mean],
                      quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        plt.show()

    plot_corner(samples)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

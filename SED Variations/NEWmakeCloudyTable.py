# written 31 May 2023 to parse *.grd files and *.linelist files and create 
# a table to be read with astropy
# v2 - fixes to correct issues with multi-parameter grids

import numpy as np
import astropy.units as u
from astropy.table import Table

def int_to_Roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while  num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

def makeCloudyTable(root,linelistsuffix='linelist.output',gridsuffix='grid.grd',write_table=False,output_suffix='_linelistgrid.txt',return_table=True,return_vac_waves=True):
    '''Reads in a Cloudy linelist output file and returns a table version of the output.

    root = the rootname of the Cloudy input file (e.g., if Cloudy was run on myscript.in
           then root='myscript')
    
    By default, this function returns the output as an Astropy Table and list of
    vacuum wavelengths, e.g.:
    
    lineListTable,vac_waves = readCloudyFiles(myscript)

    Options:
        linelistsuffix   - suffix used for Cloudy save line list command, e.g., "linelist.output" if using
                           save line list "linelist.output" "linelist_creisner.dat" intrinsic column last
 
        gridsuffix       - suffix used for Cloudy save grid command, e.g., "grid.grd" if using
                           save grid "grid.grd" last
 
        write_table      - if True, write a text file with the reformatted table (default = False)        
 
        output_suffix    - suffix to append to root if writing a text table (default = '_linelistgrid.txt')
 
        return_table     - if True, return an Astropy table with the Cloudy output (default = True)
 
        return_vac_waves - if True, return list with the vacuum wavelengths in Angstroms for each transition (default = True)

    '''
    # read in grid file
    raw_grid=open(root+gridsuffix).readlines()
    
    # find parameters for grid
    raw_cols=raw_grid[0].split()
    raw_data0=raw_grid[1].split()
    ngrid=int((len(raw_data0)-6)/2)
    grid_cols=raw_cols[7:7+ngrid]
    grid_cols=[col.strip('=') for col in grid_cols]

    # get grid param values for each iteration
    grid_params=[]
    for line in raw_grid[1:]:
        x=line.split()
        grid_iter=int(x[0])
        grid_vals=[]
        for i in range(ngrid):
            grid_vals.append(x[i+6])
        grid_params.append([grid_iter]+grid_vals)
    
    # transpose list
    grid_params=list(map(list, zip(*grid_params)))

    # create table of grid parameters
    grid_table=Table(grid_params,names=(['iter']+grid_cols))


    # read in linelist file
    raw_linelist=open(root+linelistsuffix).readlines()

    # loop through lines to find grid delimiters and remove extraneous lines/characters
    grid_of_linelists=[]
    line_ions=[]
    line_waves=[]
    line_ids=[]
    this_linelist=[]
    grid_iter=-1
    for line in raw_linelist:
        if 'GRID_DELIMIT' in line:
            # for first line, create columns for iteration, grid params, and line IDs
            if grid_iter<0:
                grid_of_linelists.append(['grid_iter']+grid_cols+line_ids)

            # determine the value of the grid parameter for this iteration
            grid_string=line.rstrip().split()[-1]
            grid_iter=int(grid_string[-9:])

            # append the iteration value, grid params, and line values to the grid
            grid_idx=np.where(np.asarray(grid_table['iter'],dtype=int)==grid_iter)
            grid_vals=[float(grid_table[col][grid_idx]) for col in grid_cols]
            grid_strs=['{:12.5f}'.format(val) for val in grid_vals]
            grid_of_linelists.append([grid_iter]+grid_strs+this_linelist)

            # clear the line list string for the next iteration
            this_linelist=[]
        elif (line[0]=='#') or (line[0]=='\n') or (line.split()[0]=='iteration'):
            continue
        else:
            x=[line[:4]]+line.rstrip().split()[-2:]
            if grid_iter<0:
                
                # create ion string with roman numerals
                raw_ion_str=x[0]
                ion_list=raw_ion_str.split()
                if len(ion_list)==1:
                    ion_str=ion_list[0].capitalize()
                elif len(ion_list)==2:
                    ion_name=ion_list[0].capitalize()
                    ion_num=int_to_Roman(int(ion_list[1]))
                    ion_str=ion_name+ion_num
                
                # create wavelength string (rounded, angstroms)
                raw_wave_str=x[1]
                wave_val_orig=float(raw_wave_str[:-2])
                wave_unit_orig=raw_wave_str[-1]
                if wave_unit_orig=='A':
                    wave_quantity=wave_val_orig*u.AA
                elif wave_unit_orig=='m':
                    wave_quantity=wave_val_orig*u.um
                else:
                    print('Wavelength unit {:s} is not recognized. Treating as Angstroms.'.format(wave_unit_orig))
                    wave_quantity=wave_val_orig*u.AA
                wave_val_angstroms=wave_quantity.to(u.AA).value
                wave_round_angstroms='{:1.0f}'.format(wave_val_angstroms)
                
                # create line ID string
                line_id_str=ion_str+'_'+wave_round_angstroms

                # calculate vacuum wavelength for sub-header
                if (wave_val_angstroms<2000) or (wave_val_angstroms>1e4):
                    wave_vac=wave_val_angstroms
                else:
                    wave_vac=wave_val_angstroms*1.000293
                wave_vac_str='{:4.2f}'.format(wave_vac)
                
                # append values to lists
                line_ions.append(ion_str)
                line_waves.append(wave_vac_str)
                line_ids.append(line_id_str)
            
            # append Hb ratio to linelist for this iteration
            this_linelist.append(x[2])

    # get column names of full table
    table_cols=grid_of_linelists[0]

    # return output in requested format
    return_items=[]

    if return_table:
        # transpose grid of lists
        table_data=list(map(list, zip(*grid_of_linelists[1:])))

        # create table
        linelistTable=Table(table_data,names=table_cols,dtype=([int]+[float]*(len(table_cols)-1)))

        # append to return_items
        return_items.append(linelistTable)

    if return_vac_waves:
        return_items.append(line_waves)
    
    if write_table:
        output_filename=root+output_suffix
        f=open(output_filename,'w')
        # table_cols_10char=['{:10s}'.format(col) for col in table_cols] 
        # table_waves_10char=['vac_wave:',' '*10]+['{:10s}'.format(wave) for wave in line_waves]
        table_cols_10char=[col.rjust(10,' ') for col in table_cols] 
        table_waves_10char=[col.rjust(10,' ') for col in (['vac_wave:']+[' ']*ngrid+line_waves)]
        f.write('# '+' '.join(table_cols_10char)+'\n')
        f.write('# '+' '.join(table_waves_10char)+'\n')
        for row in grid_of_linelists[1:]:
            iter_str='{:10.0f}'.format(row[0])
            # table_data_10char=[iter_str]+['{:10s}'.format(col) for col in row[1:]]
            table_data_10char=[iter_str]+[col.rjust(10,' ') for col in row[1:]]
            f.write(' '.join(table_data_10char)+'\n')
        f.close()
        print('Wrote file '+output_filename)

    return return_items

if __name__ == '__main__':
    import sys
    args=sys.argv
    root=args[1]
    out=makeCloudyTable(root,write_table=True,return_table=False,return_vac_waves=False)


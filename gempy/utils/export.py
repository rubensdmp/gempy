
from matplotlib.cm import ScalarMappable as SM
from gempy.plot.visualization_2d import PlotData2D
import numpy as np
import os
import gempy as gp


def export_geomap2geotiff(path, geo_model, geo_map=None, geotiff_filepath=None):
    """

    Args:
        path (str): Filepath for the exported geotiff, must end in .tif
        geo_map (np.ndarray): 2-D array containing the geological map
        cmap (matplotlib colormap): The colormap to be used for the export
        geotiff_filepath (str): Filepath of the template geotiff

    Returns:
        Saves the geological map as a geotiff to the given path.
    """
    import gdal

    plot = PlotData2D(geo_model)
    cmap = plot._cmap
    norm = plot._norm

    if geo_map is None:
        geo_map = geo_model.solutions.geological_map[0].reshape(geo_model.grid.topography.resolution)

    if geotiff_filepath is None:
        # call the other function
        print('stupid')

    # **********************************************************************
    geo_map_rgb = SM(norm=norm, cmap=cmap).to_rgba(geo_map.T) # r,g,b,alpha
    # **********************************************************************
    # gdal.UseExceptions()
    ds = gdal.Open(geotiff_filepath)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    outFileName = path
    driver = gdal.GetDriverByName("GTiff")
    options = ['PROFILE=GeoTiff', 'PHOTOMETRIC=RGB', 'COMPRESS=JPEG']
    outdata = driver.Create(outFileName, rows, cols, 3, gdal.GDT_Byte, options=options)

    outdata.SetGeoTransform(ds.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(geo_map_rgb[:, ::-1, 0].T * 256)
    outdata.GetRasterBand(2).WriteArray(geo_map_rgb[:, ::-1, 1].T * 256)
    outdata.GetRasterBand(3).WriteArray(geo_map_rgb[:, ::-1, 2].T * 256)
    outdata.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    outdata.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    outdata.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    # outdata.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)  # alpha band

    # outdata.GetRasterBand(1).SetNoDataValue(999)##if you want these values transparent
    outdata.FlushCache()  # saves to disk
    outdata = None  # closes file (important)
    band = None
    ds = None

    print("Successfully exported geological map to  " +path)

def export_moose_input(geo_model, path=None):
    """
    Method to export a 3D geological model as MOOSE compatible input. 

    Args:
        path (str): Filepath for the exported input file

    Returns:
        
    """
    # get model dimensions
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = geo_model.solutions.grid.regular_grid.extent
    
    # get unit IDs and restructure them
    ids = np.round(geo_model.solutions.lith_block)
    ids = ids.astype(int)
    
    liths = ids.reshape((nx, ny, nz))
    liths = liths.flatten('F')

    # create unit ID string for the fstring
    idstring = '\n  '.join(map(str, liths))

    # create a dictionary with unit names and corresponding unit IDs
    sids = dict(zip(geo_model.surfaces.df['surface'], geo_model.surfaces.df['id']))
    surfs = list(sids.keys())
    uids = list(sids.values())
    # create strings for fstring, so in MOOSE, units have a name instead of an ID
    surfs_string = ' '.join(surfs)
    ids_string = ' '.join(map(str, uids))
    
    fstring = f"""[MeshGenerators]
  [./gmg]
  type = GeneratedMeshGenerator
  dim = 3
  nx = {nx}
  ny = {ny}
  nz = {nz}
  xmin = {xmin}
  xmax = {xmax}
  yim = {ymin}
  ymax = {ymax}
  zmin = {zmin}
  zmax = {zmax}
  block_id = '{ids_string}'
  block_name = '{surfs_string}'
  [../]
  
  [./subdomains]
    type = ElementSubdomainIDGenerator
    input = gmg
    subdomain_ids = '{idstring}'
  [../]
[]

[Mesh]
  type = MeshGeneratorMesh
[]
"""
    if not path:
        path = './'
    if not os.path.exists(path):
      os.makedirs(path)

    f = open(path+'geo_model_units_moose_input.i', 'w+')
    
    f.write(fstring)
    f.close()
    
    print("Successfully exported geological model as moose input to "+path)


def export_surfaces_to_obj(geo_model, path):
    """Export model surfaces as obj-files that can be imported into Blender.

    Args:
        path: Filepath to which the obj-files of the surfaces are to be exported. Should end with "/".
        geo_model: Solution of the computed geomodel.

    Returns:
        One obj-file for each surface in the model.
    """
    filepath_export = path
    surface_names = geo_model.surfaces.df['surface']
    surface_colors = geo_model.surfaces.colors.colordict

    vertices_all_surfaces =  gp.get_surfaces(geo_model.solutions)[0]
    simplices_all_surfaces =  gp.get_surfaces(geo_model.solutions)[1]

    for i in np.arange(len(vertices_all_surfaces)):
        with open(filepath_export+"surface_{}.obj".format(surface_names[i]), 'w') as ofile:
            # color
            # saved as first entry in obj-file
            # is not automatically set when imported
            color = surface_colors[surface_names[i]]
            line = 'color {}'.format(color) + '\n'
            ofile.write(line)

            for v in vertices_all_surfaces[i]:
                line = "v" + " {:.5}".format(v[0]) + " {:.5}".format(v[1]) + " {:.5} ".format(v[2]) + "\n"
                line = line.format()
                ofile.write(line)

            for s in simplices_all_surfaces[i]:
                line = "{0} {1} "
                # obj vertex indices for faces start at 1 not 0
                indices = [str(i + 1) for i in s[:]]
                line = line.format("f", ' '.join(indices) + "\n")
                ofile.write(line)



"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes and methods to perform implicit regional modelling based on
    the potential field method.
    Tested on Ubuntu 16

    Created on 10/10 /2016

    @author: Miguel de la Varga
"""

from os import path
import sys

# This is for sphenix to find the packages
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as _np
from numpy import ndarray
from pandas import DataFrame
from gempy.plot.sequential_pile import StratigraphicPile
from gempy.core.model import *


# region Model level functions
def create_data(extent, resolution=(50, 50, 50), project_name='default_project', **kwargs) -> Model:

    """
    DEP
    Method to create a :class:`gempy.data_management.InputData` object. It is analogous to gempy.InputData()

    Args:
        extent (list or array):  [x_min, x_max, y_min, y_max, z_min, z_max]. Extent for the visualization of data
         and default of for the grid class.
        resolution (list or array): [nx, ny, nz]. Resolution for the visualization of data
         and default of for the grid class.

    Keyword Args:

        path_i: Path to the data bases of interfaces. Default os.getcwd(),
        path_o: Path to the data bases of orientations. Default os.getcwd()

    Returns:
        :class:`gempy.data_management.InputData`

    """
    warnings.warn("This method will get deprecated in the next version of gempy. It still exist only to keep"
                  "the behaviour equal to older version. See create_model.", FutureWarning)
    model = create_model(project_name)
    set_grid(model, create_grid(grid_type='regular_grid', extent=extent, resolution=resolution))
    read_data(model, **kwargs)
    set_values_to_default(model, series_distribution=model.interfaces, order_series = None, order_formations=None,
                          set_faults=True, map_formations_from_series=True, call_map_to_data=True, verbose=0)
    update_additional_data(model)

    return model


def set_values_to_default(model: Model, series_distribution=None, order_series=None, order_formations=None,
                          set_faults=True, map_formations_from_series=True, call_map_to_data=True, verbose=0):

    if series_distribution:
        model.series.set_series_categories(series_distribution, order=order_series)

    if set_faults is True:
        model.faults.set_is_fault()

    if map_formations_from_series is True:
        model.formations.map_formations_from_series(model.series)
        model.formations.df = model.formations.set_id(model.formations.df)
        try:
            model.formations.add_basement()
            model.series.add_basement()
        except AssertionError:
            print('already basement')
            pass
    if order_formations is not None:
        warnings.warn(" ", FutureWarning)
        model.formations.set_formation_order(order_formations)

    if call_map_to_data is True:
        map_to_data(model, model.series, model.formations, model.faults)

    if verbose > 0:
        return get_sequential_pile(model)
    else:
        return True


def map_to_data(model: Model, series: Series=None, formations: Formations=None, faults: Faults=None):
    # TODO this function makes sense as Model method
    if series is not None:
        model.interfaces.map_series_to_data(series)
        model.orientations.map_series_to_data(series)

    if formations is not None:
        model.interfaces.map_formations_to_data(formations)
        model.orientations.map_formations_to_data(formations)

    if faults is not None:
        model.interfaces.map_faults_to_data(faults)
        model.orientations.map_faults_to_data(faults)


def get_data(model: Model, itype='data', numeric=False, verbosity=0):
    """
    Method to return the data stored in :class:`DataFrame` within a :class:`gempy.interpolator.InterpolatorData`
    object.

    Args:
        model (:class:`gempy.core.model.Model`)
        itype(str {'all', 'interfaces', 'orientaions', 'formations', 'series', 'faults', 'fautls_relations'}): input
            data type to be retrieved.
        numeric (bool): if True it only returns numberical properties. This may be useful due to memory issues
        verbosity (int): Number of properties shown

    Returns:
        pandas.core.frame.DataFrame

    """
    return model.get_data(itype=itype, numeric=numeric, verbosity=verbosity)
# endregion


# region Model
def create_model(project_name='default_project'):
    return Model(project_name)


def save_model(model: Model, path=False):
    """
     Save InputData object to a python pickle (serialization of python). Be aware that if the dependencies
     versions used to export and import the pickle differ it may give problems

     Args:
         geo_data (:class:`gempy.data_management.InputData`)
         path (str): path where save the pickle (without .pickle)

     Returns:
         None
     """
    model.save_model(path)


def load_model(path):
    """
    Read InputData object from python pickle.

    Args:
       path (str): path where save the pickle

    Returns:
        :class:`gempy.data_management.InputData`

    """
    import pickle
    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        model = pickle.load(f)
        return model
# endregion


# region Series functionality
def create_series(series_distribution=None, order=None):
    return Series(series_distribution=series_distribution, order=order)


def set_series(model: Model, series_distribution, order_series=None, order_formations=None,
               values_to_default=True, verbose=0):
    """
    Function to set in place the different series of the project with their correspondent formations

    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        series_distribution (dict or :class:`DataFrame`): with the name of the series as key and the name of the
          formations as values.
        order_series(Optional[list]): only necessary if passed a dict (python < 3.6)order of the series by default takes the
             dictionary keys which until python 3.6 are random. This is important to set the erosion relations between the different series
        order_formations(Optional[list]): only necessary if passed a dict (python < 3.6)order of the series by default takes the
            dictionary keys which until python 3.6 are random. This is important to set the erosion relations between the different series
        verbose(int): if verbose is True plot hte sequential pile

    Notes:
        The following dataframes will be modified in place
            1) geo_data.series: A pandas DataFrame with the series and formations relations
            2) geo_data.interfaces: one extra column with the given series
            3) geo_data.orientations: one extra column with the given series
    """

    model.series.set_series_categories(series_distribution, order=order_series)
    if values_to_default is True:
        warnings.warn("This option will get deprecated in the next version of gempy. It still exist only to keep"
                      "the behaviour equal to older version. See set_values_to_default.", FutureWarning)

        set_values_to_default(model, order_formations=None, set_faults=True,
                              map_formations_from_series=True, call_map_to_data=True)

    update_additional_data(model)

    if verbose > 0:
        return get_sequential_pile(model)
    else:
        return None


def select_series(geo_data, series):
    """
    Return the formations of a given serie in string

    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        series(list of int or list of str): Subset of series to be selected

    Returns:
         :class:`gempy.data_management.InputData`: New object only containing the selected series
    """
    new_geo_data = copy.deepcopy(geo_data)

    if type(series) == int or type(series[0]) == int:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['order_series'].isin(series)]
        new_geo_data.orientations = geo_data.orientations[geo_data.orientations['order_series'].isin(series)]
    elif type(series[0]) == str:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['series'].isin(series)]
        new_geo_data.orientations = geo_data.orientations[geo_data.orientations['series'].isin(series)]

    # Count faults
    new_geo_data.set_faults(new_geo_data.count_faults())

    # Change the dataframe with the series
    new_geo_data.series = new_geo_data.series[new_geo_data.interfaces['series'].unique().
        remove_unused_categories().categories].dropna(how='all')
    new_geo_data.formations = new_geo_data.formations.loc[new_geo_data.interfaces['formation'].unique().
        remove_unused_categories().categories]
    new_geo_data.update_df()
    return new_geo_data


def get_series(model: Model):
    """
    Args:
         geo_data (:class:`gempy.data_management.InputData`)

    Returns:
        :class:`DataFrame`: Return series and formations relations
    """
    return model.series


def get_sequential_pile(model: Model):
    """
    Visualize an interactive stratigraphic pile to move around the formations and the series. IMPORTANT NOTE:
    To have the interactive properties it is necessary the use of an interactive backend. (In notebook use:
    %matplotlib qt5 or notebook)

    Args:
        geo_data (:class:`gempy.interpolator.InterpolatorData`)

    Returns:
        :class:`matplotlib.pyplot.Figure`
    """
    return StratigraphicPile(model.series)
# endregion


# region Formations functionality
def create_formations(values_array=None, values_names=np.empty(0), formation_names=np.empty(0)):
    f = Formations(values_array=values_array, properties_names=values_names, formation_names=formation_names)
    return f


def set_formations(geo_data: Model, formations=None, formations_order=None, formations_values=None,
                   properties_names=None):
    """
    Function to order and change the value of the model formations. The values of the formations will be the final
    numerical value that each formation will take in the interpolated geological model (lithology block)
    Args:
        geo_data (:class:`gempy.data_management.InputData`):
        formations_order (list of str): List with a given order of the formations. Due to the interpolation algorithm
            this order is only relevant to keep consistent the colors of layers and input data. The order ultimately is
            determined by the geometric sedimentary order
        formations (list of str): same as formations order. you can use any
        formations_values (list of floats or int):  values of the formations will be the final
    numerical value that each formation will take in the interpolated geological model (lithology block)

    Returns:
        :class:`DataFrame`: formations dataframe already updated in place

    """
    if formations and not formations_order:
        formations_order = formations
    if formations_order is not None and formations_values is not None:
        geo_data.formations.set_formations_values(formations_values, formation_order=formations_order,
                                                  properties_names=properties_names)
    elif formations_order is not None:
        geo_data.formations.set_formation_order(formations_order)
        geo_data.formations.set_id()

    return geo_data.formations


def set_order_formations(geo_model, formation_order):
    warnings.warn("set_order_formations will be removed in version 1.2, "
                  "use gempy.set_formations function instead", FutureWarning)
    set_formations(geo_model, formations_order=formation_order)


def get_formations(model: Model):
    return model.formations
# endregion


# region Fault functionality
def create_faults(series: Series, series_fault=None, rel_matrix=None):
    return Faults(series=series, series_fault=series_fault, rel_matrix=rel_matrix)


def set_faults(model: Model, faults: Faults):
    model.faults = faults


def get_faults(model: Model):
    return model.faults
# endregion


# region Grid functionality
def create_grid(grid_type: str, **kwargs):
    return GridClass(grid_type=grid_type, **kwargs)


def set_grid(model: Model, grid: GridClass, only_model=False):
    """

    Args:
        model (object):
    """
    model.set_grid(grid=grid, only_model=only_model)


def get_grid(model: Model):
    """
    Coordinates can be found in :class:`gempy.data_management.GridClass.values`

     Args:
          geo_data (:class:`gempy.interpolator.InterpolatorData`)

     Returns:
        :class:`gempy.data_management.GridClass`
    """
    return model.grid


def get_resolution(model: Model):
    return model.grid.resolution


def get_extent(model: Model):
    return model.grid.extent


def update_grid(model, grid_type: str, **kwargs):
    model.grid.__init__(grid_type=grid_type, **kwargs)
# endregion


# region Point-Orientation functionality
def read_data(geo_model: Model, path_i=None, path_o=None, **kwargs):
    """
    For extra options in reading please look at the Interfaces methods
    Args:
        model:
        path_i:
        path_o:
        **kwargs: pandas read table kwargs
    Returns:

    """

    if path_i:
        geo_model.interfaces.read_interfaces(path_i, inplace=True, **kwargs)
    if path_o:
        geo_model.orientations.read_orientations(path_o, inplace=True, **kwargs)

    geo_model.rescaling.rescale_data()
    update_additional_data(geo_model)


def set_interfaces(geo_data, interf_dataframe, append=False):
    """
     Method to change or append a Dataframe to interfaces in place.

     Args:
         geo_data(:class:`gempy.data_management.InputData`)
         interf_dataframe (:class:`DataFrame`)
         append (Bool): if you want to append the new data frame or substitute it
     """
    geo_data.set_interfaces(interf_dataframe, append=append)


def get_interfaces(model: Model):
    return model.interfaces


def set_orientations(geo_data, orient_dataframe, append=False):
    """
    Method to change or append a dataframe to orientations in place.  A equivalent Pandas Dataframe with
    ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation'] has to be passed.

    Args:
         geo_data(:class:`gempy.data_management.InputData`)
         interf_dataframe (:class:`DataFrame`)
         append (Bool): if you want to append the new data frame or substitute it
    """

    geo_data.set_orientations(orient_dataframe, append=append)


def set_orientation_from_interfaces(geo_data, indices_array):
    """
    Create and set orientations from at least 3 points of the :attr:`gempy.data_management.InputData.interfaces`
     Dataframe
    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        indices_array (array-like): 1D or 2D array with the pandas indices of the
          :attr:`gempy.data_management.InputData.interfaces`. If 2D every row of the 2D matrix will be used to create an
          orientation
        verbose:

    Returns:
        :attr:`gempy.data_management.InputData.orientations`: Already updated inplace
    """

    if _np.ndim(indices_array) is 1:
        indices = indices_array
        form = geo_data.interfaces['formation'].loc[indices].unique()
        assert form.shape[0] is 1, 'The interface points must belong to the same formation'
        form = form[0]
        print()
        ori_parameters = geo_data.create_orientation_from_interfaces(indices)
        geo_data.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                 dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                 G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                 formation=form)
    elif _np.ndim(indices_array) is 2:
        for indices in indices_array:
            form = geo_data.interfaces['formation'].loc[indices].unique()
            assert form.shape[0] is 1, 'The interface points must belong to the same formation'
            form = form[0]
            ori_parameters = geo_data.create_orientation_from_interfaces(indices)
            geo_data.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                     dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                     G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                     formation=form)

    geo_data.update_df()
    return geo_data.orientations


def get_orientations(model: Model):
    return model.orientations


def rescale_data(model: Model, rescaling_factor=None, centers=None):
    """
    Is always in place.

    Rescale the data of a :class:`gempy.data_management.InputData`
    object between 0 and 1 due to stability problem of the float32.

    Args:
        geo_data(:class:`gempy.data_management.InputData`)
        rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

    Returns:
        gempy.data_management.InputData: Rescaled data

    """

    model.rescaling.rescale_data(rescaling_factor, centers)
# endregion


# region Interpolator functionality
def set_interpolation_data(model: Model, inplace=True, **kwargs):
    """
    Create a :class:`gempy.interpolator.InterpolatorData`. InterpolatorData is a class that contains all the
     preprocessing operations to prepare the data to compute the model.
     Also is the object that has to be manipulated to vary the data without recompile the modeling function.

    Args:
        geo_data(gempy.DataManagement.InputData): All values of a DataManagement object
        compile_theano (bool): select if the theano function is compiled during the initialization. Default: True
        compute_all (bool): If true the solution gives back the block model of lithologies, the potential field and
         the block model of faults. If False only return the block model of lithologies. This may be important to speed
          up the computation. Default True
        u_grade (list): grade of the polynomial for the universal part of the Kriging interpolations. The value has to
        be either 0, 3 or 9 (number of equations) and the length has to be the number of series. By default the value
        depends on the number of points given as input to try to avoid singular matrix. NOTE: if during the computation
        of the model a singular matrix is returned try to reduce the u_grade of the series.
        rescaling_factor (float): rescaling factor of the input data to improve the stability when float32 is used. By
        defaut the rescaling factor is calculated to obtein values between 0 and 1.

    Keyword Args:
         dtype ('str'): Choosing if using float32 or float64. This is important if is intended to use the GPU
         See Also InterpolatorClass kwargs

    Attributes:
        geo_data: Original gempy.DataManagement.InputData object
        geo_data_res: Rescaled data. It has the same structure has gempy.InputData
        interpolator: Instance of the gempy.DataManagement.InterpolaorInput.InterpolatorClass. See Also
         gempy.DataManagement.InterpolaorInput.InterpolatorClass docs
         th_fn: Theano function which compute the interpolation
        dtype:  type of float

    """
    # TODO add kwargs
    model.rescaling.rescale_data()
    update_additional_data(model)

    model.interpolator.create_theano_graph()
    model.interpolator.set_theano_shared_parameters()

    compile_theano = kwargs.get('compile_theano', True)
    if compile_theano is True:
        model.interpolator.compile_th_fn(inplace=inplace)

    return model.interpolator


def get_interpolator(model: Model):
    return model.interpolator


def get_th_fn(model: Model):
    """
    Get the compiled theano function

    Args:
        interp_data (:class:`gempy.data_management.InputData`)

    Returns:
        :class:`theano.compile.function_module.Function`: Compiled function if C or CUDA which computes the interpolation given the input data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref interfaces, XYZ rest interfaces)
    """
    assert getattr(model.interpolator, 'theano_function', False) is not None, 'Theano has not been compiled yet'

    return model.interpolator.theano_function
# endregion


# region Additional data functionality
def update_additional_data(model: Model, update_structure=True, update_rescaling=True, update_kriging=True):
    if update_structure is True:
        model.additional_data.update_structure()
    if update_rescaling is True:
        model.additional_data.update_rescaling_data()
    if update_kriging is True:
        model.additional_data.update_default_kriging()

    return model.additional_data


def get_additional_data(model: Model):
    return model.additional_data


def get_kriging_parameters(model: Model):
    """
    Print the kringing parameters

    Args:
        interp_data (:class:`gempy.data_management.InputData`)
        verbose (int): if > 0 print all the shape values as well.

    Returns:
        None
    """
    return model.additional_data.kriging_data
# endregion


# region Computing the model
def compute_model(geo_model: Model)-> Solution:
    """
    Computes the geological model and any extra output given in the additional data option.

    Args:
        geo_model (:obj:`gempy.core.data.Model`)

    Returns:
        gempy.core.data.Solution

    """
    # TODO: Assert frame by frame that all data is like is supposed. Otherwise,
    # return clear messages
    i = geo_model.interpolator.get_input_matrix()

    assert geo_model.additional_data.len_formations_i.min() > 1,  \
        'To compute the model is necessary at least 2 interface points per layer'

    sol = geo_model.interpolator.theano_function(*i)
    geo_model.solutions.set_values(sol)

    return geo_model.solutions


def compute_model_at(new_grid_array: ndarray, geo_model: Model):
    """
    This function does the same as :func:`gempy.core.gempy_front.compute_model` plus the addion functionallity of
     passing a given array of points where evaluate the model instead of using the :class:`gempy.core.data.GridClass`.

    Args:
        geo_model:
        new_grid_array (:class:`_np.array`): 2D array with XYZ (columns) coorinates

    Returns:
        gempy.core.data.Solution
    """

    set_grid(geo_model, create_grid('custom_grid', custom_grid=new_grid_array))

    # Now we are good to compute the model again only in the new point
    sol = compute_model(geo_model)
    return sol
# endregion


# region Solution
def get_surfaces(model: Model):
    """
    Compute vertices and simplices of the interfaces for its vtk visualization and further
    analysis

    Args:
        interp_data (:class:`gempy.data_management.InputData`)
        potential_lith (ndarray): 1D numpy array with the solution of the computation of the model
         containing the scalar field of potentials (second row of lith solution)
        potential_fault (ndarray): 1D numpy array with the solution of the computation of the model
         containing the scalar field of the faults (every second row of fault solution)
        n_formation (int or 'all'): Positive integer with the number of the formation of which the surface is returned.
         use method get_formation_number() to get a dictionary back with the values
        step_size (int): resolution of the method. This is every how many voxels the marching cube method is applied
        original_scale (bool): choosing if the coordinates of the vertices are given in the original or the rescaled
         coordinates

    Returns:
        vertices, simpleces
    """
    return model.solutions.vertices, model.solutions.edges
# endregion

# region Auxiliary
def _setdoc(docstring):
    def decor(func):
        if func.__doc__ is None:
            func.__doc__ = docstring
        else:
            func.__doc__ += '/n' + docstring

        return func

    return decor
# endregion
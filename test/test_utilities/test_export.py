import pytest

import os
import gempy as gp
import numpy as np
from gempy.utils import export

path_to_data = os.path.dirname(__file__)+'/../../notebooks/'
filepath_export = os.path.dirname(__file__) + '/../../test/test_utilities/temp/'
filepath_ref = os.path.dirname(__file__) + '/../../test/input_data/'

def test_export_to_obj():

    geo_data = gp.create_data([0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                              path_o=path_to_data + "data/input_data/jan_models/model2_orientations.csv",
                              path_i=path_to_data + "data/input_data/jan_models/model2_surface_points.csv")

    gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})

    gp.set_interpolator(geo_data, theano_optimizer='fast_compile')
    geo_model = gp.compute_model(geo_data)

    export.export_surfaces_to_obj(geo_model, filepath_export)

    exported = filepath_export+'surface_rock1.obj'
    reference = filepath_ref+'surface_rock1.obj'

    exported_obj = open(exported, 'r')
    lines_exported = exported_obj.readlines()

    reference_obj = open(reference, 'r')
    lines_reference = reference_obj.readlines()

    for l in lines_reference:
        assert l == lines_exported[l]

    #self.assertMultiLineEqual(lines_exported, lines_reference)
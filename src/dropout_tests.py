import os
import shutil

def make_directory(run_name):
    '''This function checks that the relevant directory is made in data/run_name for the subdivided dropout datasets.

    :param run_name: The name of this model run.
    :type run_name: string
    '''

    try:
        os.mkdir('data/' + run_name + '/dropout')
    except FileExistsError:
        shutil.rmtree('data/' + run_name + '/dropout')
        os.mkdir('data/' + run_name + '/dropout')


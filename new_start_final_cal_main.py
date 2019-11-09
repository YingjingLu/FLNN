from data_source.baseline_mnist_dataSource import * 
import numpy as np 
import os as os
from model.new_start_final_cal import * 
from opts.new_start_final_cal_opts import *


if __name__ == "__main__":

    opts = New_Start_Final_Cal_Opts() 
    opts.data_source = Data_Source( opts )
    opts.data_source.load_unsplit_samples()

    f = New_Start_Final_Cal( opts )
    if opts.train:
        f.train()

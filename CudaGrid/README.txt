Assignment: Final Project
Course: AM 148
Programmer: Bryan Garcia

Description: A gpu implementation of bicubic interpolation for a regularly spaced grid

Contents:
    
    CudaGrid : directory containing source and makefile

        CudaGrid.* : Header file for main class handling bicubic interpolation, 
                     and implementation of header file. *Contains the GPU kernels*

        psuedo_ND.* : Source for my ndarray data structure class. Relied upon heavily
                      in CudaGrid class

        CudaGrid_Driver.* : Interface for calling CUDA code from Python.

        plink.py : Python script for demoing the bicubic interpolation on the GPU.

        Bicubic.* : Independent bicubic class intended to be plugged into CudaGrid class.
                    * Not utilized but included since mentioned in "report." *



To run:
    
    -- As a job...

        The makefile is setup with the recipes required to compile the code. If on Lux,
        you can submit a job with 'make pbatch'. 

        Required lux modules:

            cuda10.1/
            python
            numpy
            matplotlib

    -- Interactively...

        1. Create the shared library, run "make lib".

        2. Set the SO_PATH variable to the current directory
            - I've been doing `export SO_PATH=$(pwd)'

        3. run 'python plink.py'





    


    

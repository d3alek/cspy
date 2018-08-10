# CSPy

## Get latest version

git clone https://github.com/d3alek/cspy

## Install requirements in a virtualenv

```
> virtualenv venv
> source venv/bin/activate
> pip install -r requirements.txt
```

## Run
Make sure you have ran `source venv/bin/activate` previously in the current console.

```
> python cspy.py <mat-file>
```

For example:

> python cspy.py RRptakPLx400.mat

You will then be asked to write a number representing the table in that mat file to fit. If everything works, this will show 3 graphs - one with the peaks detected by `update_spec_from_peaks` function, another with the fit and residuals, and a third one with the gaussian components of the fit. You will also get the best fit parameters for each gaussian in the output.

Optionally, you can provide number of gaussians (defaults to 0) and their width (defaults to 50):

> python cspy.py RRptakPLx400.mat -g 3 -w 100

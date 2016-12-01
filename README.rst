Peter Duggins, Terry Stewart, Xuan Choo, Chris Eliasmith

Effects of Guanfacine and Phenylephrine on a Spiking Neuron Model of Working Memory

Install
============

Clone the GitHub repository
---------------------------
1. Install Python 2.7.X (https://www.python.org/downloads/) and Git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
2. Open a terminal (Max/Linux) or a command prompt (Windows)
3. Clone the repository:
	- git clone https://github.com/psipeter/drugs_and_working_memory.git
4. Install with Pip (OR python):
	- cd drugs_and_working_memory
	- pip install .
	- (python setup.py develop)
5. Install missing packages if necessary
	- The SciPy Stack: Matplotlib, Numpy, Scipy, and Pandas (https://www.scipy.org/install.html)
	- Seaborn (https://stanford.edu/~mwaskom/software/seaborn/installing.html)
	- Pathos (https://pypi.python.org/pypi/pathos) and necessary requirements

What are these files?
=====================
1. 'model.py' is the main file for running the simulation
2. 'helper.py' has supporting functions and classes for model.py
3. 'gui.py' is for loading the model into nengo_gui and running individual simulations
4. 'remake_plots.py' is for replotting data pickled and exported by model.py
5. 'integrator_demo.py' generates the data for Figure 2
6. 'parameters.txt' specifies all of those great parameters
7. 'plots' folder contains the plots used in the paper and associated parameter files
8. 'data' folder stores the outputs of model.py, including parameters.json, plots.png, and data.pkl

Run
=======

1. Navigate to the 'rugs_and_working_memory/rugs_and_working_memory' folder in the terminal/cmd
2. Edit the 'parameters.txt' file
	- Be sure to reset the seed to get unique behavior.
3. run 'python model.py'
4. You should see the message 'Running [simulation details]...' followed by a progress indicator
5. When the simulation finishes, check out the 'data' folder to see your results. 

Questions?
==========
1. Email 'psipeter@gmail.com'

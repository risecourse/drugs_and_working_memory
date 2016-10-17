from setuptools import setup

def check_dependencies():
    install_requires = []
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    try:
        import seaborn
    except ImportError:
        install_requires.append('seaborn')
    try:
        import pathos
    except ImportError:
        install_requires.append('pathos')
    return install_requires

def readme():
    with open('README.rst') as f:
        return f.read()

if __name__ == "__main__":
	install_requires = check_dependencies()
	setup(
		name='drugs_and_working_memory',
		version='1.0',
		description='Effects of Guanfacine and Phenylephrine on a Spiking Neuron Model of Working Memory',
		url='https://github.com/psipeter/drugs_and_working_memory',
		author='Peter Duggins, Terry Stewart, Xuan Choo, Chris Eliasmith',
		author_email='psipeter@gmail.com',
		packages=['drugs_and_working_memory'],
		long_description=readme(),
		install_requires=install_requires,
		include_package_data=True,
		zip_safe=False
	)
# Project Title

The EntDetector package offers a set of functions to analyze the presence of entanglement in quantum states described by a state vector or a density matrix. Apart from the existing methods of entanglement detection, such as Negativity, Positive Partial Transpose or Schmidt decomposition for vector states. EntDetector also offers combined methods of spectral distribution with Schmidt distribution. A method for computing gramian for quantum states is also offered. The primary work environment for the package is the Python ecosystem.

## Getting Started

Getting started with the package can be reduced to downloading entdetector.py from the repository. It is the main package file that contains a set of functions to perform the tasks set before the package. This file can be downloaded directly or by cloning the repository, e.g.:

```
git clone https://github.com/qMSUZ/EntDetector.git
```

Its dedicated installation is also not needed, the file is attached to the current script created in Python using the import keyword. This has been shown in the examples presented below.

We also recommend checking out the examples that show how to use the package.

### Prerequisites

The package requires the NumPy and Scikit packages. Naturally, it is worth using the Matplotlib package, it is used to create graphs in the given examples.

### Installing

The installation process is limited to getting the entdetector.py file only. This file can be downloaded directly from the repository, or by cloning the repository or downloading the current release.

## Features

The most important features and advantages of the EntDetector package include:

* using Python as the basic working environment of the package,
* use of NumPy and SciKit packages,
* very simple API based on functions that perform specific tasks,
* work with vector states and density matrices (support for sparse matrices is also planned in order to save memory and increase the efficiency of calculations), 
* currently known methods of entanglement detection,
* entropy value calculations,
* calculating gramians for quantum states,
* partial trace and partial transposition operation,
* spectral decomposition and Schmidt decomposition for quantum states,
* algorithm of systematic quantum state search in search of entanglement by dividing the register into two partitions,
* support for the concept of majorization of quantum states,
* the package is also easily expandable with new functions.


## Examples

Performing the Schmidt distribution on the entangled EPR state: $\mket{\psi}=\frac{1}{\sqrt{2}}\left(\mket{00}+\mket{11}\right)$ requires us to create the appropriate object, then we need to decompose. It is also worth checking if we got two Schmidt numbers, which was done as follows:


```
from entdetector import *

q = create_qubit_bell_state()
schmidt_shp=(2, 2)
s,e,f = schmidt_decomposition_for_vector_pure_state(q,schmidt_shp)
```


The package has functions for creating known states expressed as density matrices:


```
from entdetector import *

qA = create_bes_horodecki_24_state(0.5)
qB = create_bes_horodecki_33_state(0.25)
```

And also as state vectors:

```
from entdetector import *

q1 = create_ghz_state(2, 3)
q2 = create_wstate(2)
```

For the function creating the GHZ state, the first parameter is d, i.e. the dimension of a single unit of quantum information, while their number is determined by the second parameter.

The package also allows you to easily make Schmidt decomposition and spectral decomposition (SaSD), e.g. the following code:

```
from entdetector import *
d=2
p=0.55
qden = create_isotropic_qubit_state(p)
schmidt_shape = (2, 2)
e,s = create_sas_table_data(qden, schmidt_shape)
print_sas_table(e, s)

[0.7071067811865475, 0.7071067811865475] | 0.6624999999999999
[0.7071067811865475, 0.7071067811865475] | 0.11250000000000003
[1.0, 0] | 0.11249999999999999
[1.0, 0] | 0.112499999999999
```

creates a SaSD table for an isotropic state with parameter p = 0.55.

## Authors

* **Marek Sawerwain** - *Initial work* - [qMSUZ](https://github.com/qMSUZ)
* **Joanna Wiśniewska** - *Additional coding*
* **Marek Wróblewski** - *Additional coding*
* **Romang Gielerak** - *Mathematical concepts*

See also the list of [contributors](https://github.com/qMSUZ/EntDetector/Contributors) who participated in this project.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

Our thanks go to all people who wanted to use our package, and we also thank you for all votes and comments about the package.
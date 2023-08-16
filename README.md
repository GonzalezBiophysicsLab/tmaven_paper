# tmaven_paper

This repository holds the code for running the investigations presented in the paper:

Verma, A.R., Ray, K., Bodick, M., Kinz-Thompson, C.D., Gonzalez, Jr., R.L. (2023) Accurately Capturing Heterogeneity in Single-molecule Experiments with tMAVEN. In preparation.

This research uses the [tMAVEN program](https://github.com/GonzalezBiophysicsLab/tmaven). The `tMAVEN` software is separate from the investigations presented in this work, however you will need the software in order to run these. Make sure that you install `tMAVEN` so that it is in your Python PATH variable (_i.e._, this probably means installing it through [script-based approach](https://gonzalezbiophysicslab.github.io/tmaven/install.html) instead of using a binary installer).


## Installation

```bash
git clone https://github.com/GonzalezBiophysicsLab/tmaven_paper.git
```

## Usage

```bash
cd tmaven_paper
python run_paper.py <dataset> <change> <model>
```

where
- `dataset` is `hom` (Homogenous), `hetstat` (Static), or `hetdyn` (Dynamic),
- `change` is `fixed` (for Homogenous), `changeN` (for Homogenous), `changeT` (for Homogenous), `changeprop` (for Static), or `changerate` (for Dynamic),
- `model` is `composite`, `global`, or `hhmm`.

If you put an invalid combination of dataset and change, the code will output Invalid combination.

Use the help for more information on arguments required for each mode.
```bash
python run_paper.py --help
```

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)
 

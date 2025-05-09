# CogSci2025-CountingGame
Companion repository for the 2025 article "The Cognitive Complexity of Rule Changes" published in the proceedings of the 47th Annual Meeting of the Cognitive Science Society.

## Outline

<pre>
CogSci2025-CountingGame/
├─── analysis/    # Python scripts
│   ├─── plots/    # Stored plots
│   │   └─── general_accuracy_rt.pdf    # Generated plot using plots.py (Fig. 2)
│   ├─── analysis.py    # All statistical analysis
│   ├─── modeling.py    # Predictive modeling using task complexity features
│   └─── plots.py    # Plotting script for Fig. 2
│
└─── data/    # Contains data used for analysis
│   ├─── full_data.csv    # Dataset obtained from the condutcted study
│   └───  trial_info.csv    # Descriptions of presented trials in the conducted study
│
└─── LICENSE    # MIT License file
└─── README.md    # This file
└─── requirements.txt    # List of Python dependencies
</pre>

## Requirements

- Python 3.12+
  - matplotlib
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - statsmodels
  - tabulate
 
Install all dependencies via:
```bash
pip install -r requirements.txt
```

## Usage

To run any of the scripts, use:
```bash
cd /path/to/repository/analysis/
python analysis.py    # Run statistical analysis
python modeling.py    # Run modeling function
python plots.py       # Generate plot (Fig. 2), stored in ./plots/
```

## Reference

Todorovikj, S., Brand, D., & Ragni, M. (in press). The Cognitive Complexity of Rule Changes. In *Proceedings of the 47th Annual Meeting of the Cognitive Science Society*.


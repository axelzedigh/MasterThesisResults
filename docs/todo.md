# Todos
## Now
- Histogram subplot (1 row for a case, e.g. additive_noise_id=1/distance=15)(column per device)
  - Shows the approximate normal distribution of the termination points
- History log subplot (row for each additive noise case, column for each case)
- Refactor trace conversion (label_lastround) into more generalizeable
- Make a short pdf with the most interesting plots/tables as a presentation
- Plot all zedigh_2021 traces (both envs)
  - termination point
  - width
  - depth
- Get trace metadata big hall to database.
- Hypotheses test functions
  - Define H0, H1, operator, confidence=0.95, error=0.8
- Decide more additive noise method ids:
  - Collected 12.5
  - Collected 25, mean_adjusted

---

## Database
- Create queries/views for trace-metrics vs avg(termination_points)
- SNR-table (test_trace_id, SNR)
- Update rank_test__grouped.md (include trace process id)

## Storage
- Backup raw-data:
  - Cloud (google drive)
  - Seagate

## Python
- setuptools
- sphinx

## Additive noise
- Plot recorded noise (both environments).
- UML diagram of processing-flow

## Denoising
- Do Weiner-filter
  - Figure out the "right" level of noise-power
  - Decide 2-3 levels of weiner filter, add to denoising-methods
- Do CDAE-filter

## Trace analysis
- What metrics to describe traces?
  - Mean, max, min, RMS, std, var, SNR_{(RMS/RMS)**2}, SNR_{(µ)**2/∂}
  - Scaling-factor, translation
- Create function for averaging RMS and joining rank_test and trace-metadata tables using pandas
- Find resolution of USRP (dynamic level step (e.g. level-size))
  - connect dynamic range ~ resolution (number of possible features)

## Quality/control check data
- Termination point check: 0.5 or 0?
- Understand scripts
  - Label...normalizing
  - SNR
  - sc-experiment
- Noise from which device?

## Data collecting
- Collect 2.5m, device 8
- Collect 2.5m device 10 again (compare difference between measurements)
- Device 9..

## Training
- Refactor training notebook script
  - Move from notebook to a regular py-file?
- Rescale traces to maxmin for the range (204:314) (create third test_trace_id)
  - Training traces
  - Test traces
    - Try classifying with additive methods None, 3, 4, 5, 

## Visualization
- Store each plot in docs/figures/png
- Save plot-functions to a plot.py file
  - Termination-point figs
  - SNR figs
  - Trace metadata width
  - Trace metadata depth
- plot history-logs for models
- Noise visualization:
  - Noise levels of additive noise:
    - Gaussian {1...5} vs training trace depth
    - Collected {1...5} vs training trace depth
    - Rayleigh {1...5} vs training trace depth
      - RMS, 

## Statistics
- Hypothesis functions
  - chi2
  - Student t-test
- Averaging std, snr etc. correctly (update plot functions.)

## Writing
- Remove all part from the final latex doc.
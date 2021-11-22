# Todos
## Now
- Drop avg_rank column in rank_test, replace with trace_processing_id
  - Assign trace_process_id 3 to all previous analyses
  - Assign trace_process_id 4 to maxmin traces

---

## Database
- Create queries/views for trace-metrics vs avg(termination_points)
- Create table with noise_data (additive_noise_id, scale, rms)
- SNR-table (test_trace_id, SNR)

## Storage
- Backup raw-data:
  - Cloud (google drive)
  - Seagate

## Rank test analysis
- What termination criterion is currently (0 or 0.5)?

## Python
- setuptools
- numba (jit, to speed up numpy processes)
  - termination-point analysis?

## Additive noise
- Additive noise functions
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
- How to average std?
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
- Make a short pdf with the most interesting plots/tables as a presentation
- Remove all part form the final latex doc.
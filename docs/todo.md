# Todos
## Database
- Rewrite trace_metadata_width_to_sb function to only insert training/trace-process=3 once (not for 5 times)
- Remove duplicates from trace_metadata_width
    - create query for viewing duplicates
    - create query for deleting duplicates
- Remove distance, device from training_traces_width (set to None)
- Consolidate rank_test, trace_metadata_depth and trace_metadata_width into 1 database-file.
- Create queries/views for trace-metrics vs avg(termination_points)
- Create table with noise_data (additive_noise_id, scale, rms)
- SNR-table (test_trace_id, SNR)

## Rank test analysis
- What termination criterion is currently (0 or 0.5)?

## Denoising
- Do Weiner-filter
- Do CDAE-filter

## Trace analysis
- Find resolution of USRP (dynamic level step (e.g. level-size))
  - connect dynamic range ~ resolution (number of possible features)
- How to average std?

## Data collecting
- Collect 2.5m, device 8
- Collect 2.5m device 10 again (compare difference between measurements)
- Collect 5m device 10 (same angle as previously papers)
  - Try to attack. If successful recollect for devices 8, 9
  - Collect traces from 

## Training
- Rescale training and test traces to maxmin for the range (204:314) (create third test_trace_id)
  - Try classifying with additive methods None, 

## Visualization
- Store each plot in docs/figures/png
- Save plot-functions to a plot.py file

## Statistics
- Hypothesis functions
  - chi2
  - Student t-test

## Writing
- Make a short pdf with the most interesting plots/tables as a presentation
- Remove all part form the final latex doc.
# Todos
## Now
- Create queries/views for trace-metrics vs avg(termination_points)
- Collect data from big hall:
  - Noise, 2.5m, 5m
- Collect 2.5m, device 8
- Collect 2.5m device 10 again (compare difference between measurements)
- Drop avg_rank column in rank_test, replace with trace_processing_id
  - Assign trace_process_id 3 to all previous analyses
  - Assign trace_process_id 4 to 

---

## Database
- Create table with noise_data (additive_noise_id, scale, rms)
- SNR-table (test_trace_id, SNR)

## Storage
- Backup raw-data:
  - Cloud (google drive)
  - USB 64GB
  - Seagate

## Rank test analysis
- What termination criterion is currently (0 or 0.5)?

## Denoising
- Do Weiner-filter
  - Figure out the "right" level of noise-power
  - Decide 2-3 levels of weiner filter, add to denoising-methods
- Do CDAE-filter

## Trace analysis
- Find resolution of USRP (dynamic level step (e.g. level-size))
  - connect dynamic range ~ resolution (number of possible features)
- How to average std?

## Data collecting

## Training
- Rescale training and test traces to maxmin for the range (204:314) (create third test_trace_id)
  - Try classifying with additive methods None, 

## Visualization
- Store each plot in docs/figures/png
- Save plot-functions to a plot.py file
- plot history-logs for models

## Statistics
- Hypothesis functions
  - chi2
  - Student t-test
- Averaging std, snr etc. correctly (update plot functions.)

## Writing
- Make a short pdf with the most interesting plots/tables as a presentation
- Remove all part form the final latex doc.
# Todos
## Now

---

## Database
- Create queries/views for trace-metrics vs avg(termination_points)
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
- Collect data from big hall:
  - Noise, 2.5m, 5m

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

## Writing
- Make a short pdf with the most interesting plots/tables as a presentation
- Remove all part form the final latex doc.
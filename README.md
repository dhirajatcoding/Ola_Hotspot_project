## Ola Gig Hotspot Predictor

A machine learning system that helps Ola ride-hailing drivers identify 
the best zones to position themselves at any given hour to minimize idle 
wait time and maximize ride demand.

### Problem
Drivers waste significant time and fuel repositioning themselves between 
rides without any data-driven guidance on where demand is actually high.

### What It Does
Given a date and hour, the system predicts — for every zone in the city — 
the expected number of ride requests (demand) and the likely wait time 
before a driver gets picked up. It then combines both into a single 
hotspot score and ranks all zones from best to worst.

### How It Works
Raw Ola booking data is cleaned, filtered to successful rides only, and 
aggregated from individual trips into hourly zone-level summaries. 
Features are engineered at two levels: temporal features (hour of day, 
day of week, weekend flag, time block) encoded using sine/cosine 
transformation to preserve their circular nature, and lag features 
(ride counts from 1 hour ago, 2 hours ago, and 24 hours ago, plus 3-hour 
and 24-hour rolling averages) computed per zone to capture recent demand 
momentum. Wait time is approximated from inter-arrival gaps between 
consecutive bookings within each zone.

Two separate XGBoost regression models are trained — one predicting 
wait time, one predicting demand — on a time-based 80/20 train-test 
split with 5-fold time-series cross-validation. Zone rankings are 
produced by the formula: score = predicted_demand / (predicted_wait + 1).

### Tech Stack
Python · Pandas · NumPy · XGBoost · scikit-learn · Matplotlib · Joblib

### Key Results
| Metric | Wait Time Model | Demand Model |
|---|---|---|
| MAE | ~3.2 min | ~2.1 rides/hr |
| R² | ~0.847 | ~0.831 |


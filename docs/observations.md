# Observations

## Plots datasets

### Plantar_activity (insoles.csv)

- `Time`
- `left pressure 1[N/cm²]` … `left pressure 16[N/cm²]`
- `right pressure 1[N/cm²]` … `right pressure 16[N/cm²]`
- `left acceleration X[g]`, `left acceleration Y[g]`, `left acceleration Z[g]`
- `left angular X[dps]`, `left angular Y[dps]`, `left angular Z[dps]`
- `right acceleration X[g]`, `right acceleration Y[g]`, `right acceleration Z[g]`
- `right angular X[dps]`, `right angular Y[dps]`, `right angular Z[dps]`
- `left total force[N]`, `right total force[N]`
- `left center of pressure X[-0.5...+0.5]`, `left center of pressure Y[-0.5...+0.5]`, `right center of pressure X[-0.5...+0.5]`, `right center of pressure Y[-0.5...+0.5]`

#### Classification

- `Time` tracks the temporal axis of each sample.
- Pressure grid columns capture activation for each of the 16 sensors per foot.
- IMU groups (acceleration and angular velocity) describe motion at the foot level.
- Total force columns summarize net load per foot.
- Center of pressure X/Y pair describes the balance point in the [-0.5, 0.5] planar range.

### Events annotations (classif.csv)

- `Name` (action label)
- `Class` (numeric class identifier)
- `Frame Start`, `Timestamp Start`
- `Frame End`, `Timestamp End`

# Sample CSV Format for Training

## Required Column Names

Your CSV file **must** include these exact column names (case-sensitive):

### Planet Parameters
- `pl_orbper` - Orbital Period (days)
- `pl_trandurh` - Transit Duration (hours)
- `pl_trandep` - Transit Depth (parts per million)
- `pl_rade` - Planet Radius (Earth radii)
- `pl_insol` - Insolation Flux (Earth flux)
- `pl_eqt` - Equilibrium Temperature (Kelvin)
- `pl_radeerr1` - Planet Radius Upper Uncertainty

### Stellar Parameters
- `st_tmag` - TESS Magnitude
- `st_dist` - Distance to Star (parsecs)
- `st_teff` - Stellar Effective Temperature (Kelvin)
- `st_logg` - Stellar Surface Gravity (log g)
- `st_rad` - Stellar Radius (Solar radii)

### Disposition & Label
- `tfopwg_disp` - TFOP Working Group Disposition
- `label_planet` - Target Label (0 = False Positive, 1 = Confirmed Planet)

---

## Sample CSV Template

```csv
# TESS Exoplanet Training Data
# Comments starting with # are allowed and will be ignored
tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,pl_rade,pl_insol,pl_eqt,st_tmag,st_dist,st_teff,st_logg,st_rad,pl_radeerr1,label_planet
PC,3.52,2.5,1200,2.1,180.5,550,12.3,145.2,5800,4.5,1.1,0.15,1
FP,5.23,3.1,850,1.8,95.3,480,13.1,220.5,5500,4.4,0.95,0.12,0
PC,8.67,4.2,2100,3.5,320.8,680,11.8,98.7,6200,4.6,1.25,0.22,1
FP,2.15,1.8,450,1.2,55.2,420,14.2,310.4,5200,4.3,0.88,0.08,0
PC,12.45,5.5,1800,2.9,210.4,590,12.7,175.3,5950,4.5,1.15,0.18,1
```

---

## Column Descriptions

### tfopwg_disp
- **Description**: TESS Follow-up Observing Program Working Group disposition
- **Common Values**:
  - `PC` = Planet Candidate
  - `CP` = Confirmed Planet
  - `FP` = False Positive
  - `APC` = Ambiguous Planet Candidate
- **Type**: String
- **Example**: `PC`, `FP`, `CP`

### pl_orbper
- **Description**: Orbital period of the planet
- **Units**: Days
- **Type**: Float
- **Range**: 0.5 - 1000+ days
- **Example**: `3.52`, `8.67`, `12.45`

### pl_trandurh
- **Description**: Duration of the transit event
- **Units**: Hours
- **Type**: Float
- **Range**: 0.5 - 10+ hours
- **Example**: `2.5`, `3.1`, `5.5`

### pl_trandep
- **Description**: Depth of the transit (fractional decrease in brightness)
- **Units**: Parts per million (ppm)
- **Type**: Float
- **Range**: 100 - 50000 ppm
- **Example**: `1200`, `850`, `2100`

### pl_rade
- **Description**: Planet radius
- **Units**: Earth radii (R⊕)
- **Type**: Float
- **Range**: 0.5 - 20+ R⊕
- **Example**: `2.1`, `1.8`, `3.5`

### pl_insol
- **Description**: Insolation flux received by planet
- **Units**: Earth flux (S⊕)
- **Type**: Float
- **Range**: 0.1 - 10000+ S⊕
- **Example**: `180.5`, `95.3`, `320.8`

### pl_eqt
- **Description**: Planet equilibrium temperature
- **Units**: Kelvin (K)
- **Type**: Float
- **Range**: 100 - 3000+ K
- **Example**: `550`, `480`, `680`

### st_tmag
- **Description**: TESS magnitude of the host star
- **Units**: Magnitude
- **Type**: Float
- **Range**: 6 - 16 (fainter = larger number)
- **Example**: `12.3`, `13.1`, `11.8`

### st_dist
- **Description**: Distance from Earth to the star
- **Units**: Parsecs (pc)
- **Type**: Float
- **Range**: 10 - 1000+ pc
- **Example**: `145.2`, `220.5`, `98.7`

### st_teff
- **Description**: Stellar effective temperature
- **Units**: Kelvin (K)
- **Type**: Float
- **Range**: 3000 - 10000+ K
- **Example**: `5800`, `5500`, `6200`

### st_logg
- **Description**: Stellar surface gravity
- **Units**: log(cm/s²)
- **Type**: Float
- **Range**: 3.5 - 5.0
- **Example**: `4.5`, `4.4`, `4.6`

### st_rad
- **Description**: Stellar radius
- **Units**: Solar radii (R☉)
- **Type**: Float
- **Range**: 0.5 - 3+ R☉
- **Example**: `1.1`, `0.95`, `1.25`

### pl_radeerr1
- **Description**: Upper uncertainty on planet radius measurement
- **Units**: Earth radii (R⊕)
- **Type**: Float
- **Range**: 0.01 - 1+ R⊕
- **Example**: `0.15`, `0.12`, `0.22`

### label_planet
- **Description**: Training label for classification
- **Values**:
  - `0` = False Positive (not a real planet)
  - `1` = Confirmed Planet (real exoplanet)
- **Type**: Integer
- **Required**: Yes (this is your target variable)

---

## Data Format Requirements

### General Rules
1. **Comma-separated values** (CSV format)
2. **First row must be headers** with exact column names
3. **No missing column names** - all 14 columns required
4. **Comments allowed** - lines starting with `#` are ignored
5. **Decimal numbers** use period (`.`) not comma
6. **No quotes** needed around values (unless values contain commas)

### Missing Values
- Missing values can be represented as empty cells or `NaN`
- The imputation step will handle missing values during training
- Example: `PC,3.52,,1200,2.1,180.5,550,12.3,145.2,5800,4.5,1.1,0.15,1`

### Header Row
The first non-comment line must be:
```csv
tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,pl_rade,pl_insol,pl_eqt,st_tmag,st_dist,st_teff,st_logg,st_rad,pl_radeerr1,label_planet
```

---

## Example with Comments

```csv
# TESS Exoplanet Classification Training Dataset
# Source: NASA Exoplanet Archive + TFOP
# Generated: 2024-12-15
#
# Column definitions:
# - tfopwg_disp: TFOP disposition
# - pl_*: Planet parameters
# - st_*: Stellar parameters
# - label_planet: 0=FP, 1=Planet
#
tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,pl_rade,pl_insol,pl_eqt,st_tmag,st_dist,st_teff,st_logg,st_rad,pl_radeerr1,label_planet
PC,3.5214,2.48,1198.5,2.08,180.2,548.3,12.34,145.67,5812,4.52,1.09,0.147,1
FP,5.2341,3.12,847.2,1.82,95.8,482.1,13.08,221.34,5523,4.41,0.947,0.119,0
PC,8.6712,4.18,2089.3,3.47,318.5,677.9,11.76,99.23,6187,4.58,1.243,0.218,1
# This entry has missing pl_trandurh value
PC,4.1234,,1567.8,2.54,156.3,521.7,12.91,167.45,5691,4.48,1.12,0.165,1
FP,2.1543,1.79,453.6,1.23,56.1,423.8,14.21,308.92,5234,4.36,0.891,0.082,0
```

---

## Validation Checklist

Before uploading your CSV:

- [ ] File is in CSV format (`.csv` extension)
- [ ] First non-comment line contains all 14 column names
- [ ] Column names match exactly (case-sensitive)
- [ ] No extra columns or missing columns
- [ ] `label_planet` column contains only 0 or 1
- [ ] Numeric columns contain valid numbers (or are empty for missing values)
- [ ] File size is under 100 MB
- [ ] Comments (if any) start with `#`

---

## Common Errors

### "Missing required columns"
**Problem**: CSV doesn't have all required column names
**Solution**: Ensure your CSV header has all 14 columns with exact names

### "Invalid data type"
**Problem**: Non-numeric value in numeric column
**Solution**: Check that all planet/stellar parameters are numbers

### "Empty file" / "No data rows"
**Problem**: CSV has only headers or comments
**Solution**: Add data rows below the header

---

## Data Sources

This format is based on the **NASA Exoplanet Archive** standard columns:
- **TESS**: Transiting Exoplanet Survey Satellite data
- **TFOP**: TESS Follow-up Observing Program
- **NEA**: NASA Exoplanet Archive

You can download real data from:
- https://exoplanetarchive.ipac.caltech.edu/
- TESS mission data releases
- TFOP working group publications

---

## Quick Start

1. Download this template
2. Replace sample data with your actual observations
3. Ensure all 14 columns are present
4. Save as `.csv` file
5. Upload to training interface
6. Preview data to verify format
7. Configure model and train

---

**Need Help?**
- Check column names are exactly as specified
- Verify numeric columns contain numbers
- Ensure label_planet is 0 or 1
- Remove any extra columns not in the required list

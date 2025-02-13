# Healthcare Costs Dataset Documentation

## Dataset Overview
- **Filename**: healthcare_costs.csv
- **Total Records**: 7,582
- **Features**: 14 columns
- **Format**: CSV (Comma Separated Values)
- **Encoding**: UTF-8

## Column Descriptions

### Identifiers
- `X`: Unique identifier for each record (Integer)

### Demographic Features
- `age`: Age of the individual (Integer)
- `gender`: Gender of the individual (String: "male"/"female")
- `location`: State of residence (String)
- `location_type`: Type of residential area (String: "Urban"/"Country")
- `education_level`: Educational qualification (String)

### Health Indicators
- `bmi`: Body Mass Index (Float)
- `hypertension`: Hypertension status (Float: 0.0/1.0)
- `smoker`: Smoking status (String: "yes"/"no")

### Lifestyle Features
- `exercise`: Exercise activity level (String: "Active"/"Not-Active")
- `yearly_physical`: Whether individual takes yearly physical examination (String: "Yes"/"No")
- `children`: Number of children/dependents (Integer)
- `married`: Marital status (String: "Married"/"Not_Married")

### Target Variable
- `cost`: Healthcare costs (Integer)

## Data Types Summary
- **Integer Fields**: X, age, children, cost
- **Float Fields**: bmi, hypertension
- **String Fields**: smoker, location, location_type, education_level, yearly_physical, exercise, married, gender

## Usage Notes
1. Missing values may exist in 'bmi' and 'hypertension' columns
2. Categorical variables need encoding before model training
3. Cost variable represents the target for prediction
4. Location data is limited to specific states in the United States

## Data Quality Considerations
- Check for missing values in BMI and hypertension fields
- Verify consistency in categorical variables
- Ensure cost values are reasonable and within expected ranges
- Validate relationships between health indicators

## Preprocessing Requirements
1. Convert categorical variables to numeric format
2. Handle any missing values
3. Consider scaling numerical features
4. Validate and clean outliers if necessary

## Additional Notes
- The dataset contains sensitive health information and should be handled according to relevant privacy regulations
- All monetary values (cost) are in USD
- Age values represent adult population

# # Number of rows
# import numpy as np
# import pandas as pd
# num_rows = 2_000_000

# # Generating random numeric data
# heart_rate = np.random.randint(50, 120, num_rows)
# blood_oxygen = np.random.uniform(85, 100, num_rows)
# stress_level_hrv = np.random.uniform(0.5, 10, num_rows)
# bmi = np.random.uniform(15, 40, num_rows)
# age = np.random.randint(18, 90, num_rows)
# height = np.random.randint(140, 200, num_rows)
# weight = np.random.randint(40, 150, num_rows)
# glasses_per_day = np.random.randint(1, 12, num_rows)
# mental_health_anxiety = np.random.randint(1, 6, num_rows)
# mental_health_depression = np.random.randint(1, 6, num_rows)
# stress_level = np.random.randint(1, 6, num_rows)
# job_hours = np.random.randint(4, 16, num_rows)

# # Categorical data generation
# smoking_habits = np.random.choice(["Never", "Occasionally", "Daily"], num_rows)
# drinking_habits = np.random.choice(["Never", "Occasionally", "Daily"], num_rows)
# exercise_routine = np.random.choice(["Sedentary", "Light Activity", "Moderate Activity", "Heavy Activity"], num_rows)
# diet_fast_food = np.random.choice(["Yes", "No"], num_rows)
# diet_fruits_veggies = np.random.choice(["Yes", "No"], num_rows)
# diet_caffeine = np.random.choice(["Yes", "No"], num_rows)
# diet_water = np.random.choice(["Yes", "No"], num_rows)
# diet_red_meat = np.random.choice(["Yes", "No"], num_rows)
# work_type = np.random.choice(["Desk Job", "Physical Work", "Mixed"], num_rows)

# # Binary medical history data
# medical_diabetes = np.random.choice(["Yes", "No"], num_rows)
# medical_hypertension = np.random.choice(["Yes", "No"], num_rows)
# medical_heart_disease = np.random.choice(["Yes", "No"], num_rows)
# medical_cancer = np.random.choice(["Yes", "No"], num_rows)

# # Family history
# family_heart_disease = np.random.choice(["Yes", "No"], num_rows)
# family_cancer = np.random.choice(["Yes", "No"], num_rows)
# family_diabetes = np.random.choice(["Yes", "No"], num_rows)

# # Medication use
# med_hypertension = np.random.choice(["Yes", "No"], num_rows)
# med_diabetes = np.random.choice(["Yes", "No"], num_rows)
# med_painkillers = np.random.choice(["Yes", "No"], num_rows)

# # Sleep deprivation
# sleep_deprivation = np.random.choice(["Yes", "No"], num_rows)

# # Disease outputs (binary classification)
# heart_disease = np.random.choice(["Yes", "No"], num_rows)
# hypertension = np.random.choice(["Yes", "No"], num_rows)
# stroke = np.random.choice(["Yes", "No"], num_rows)
# diabetes = np.random.choice(["Yes", "No"], num_rows)
# obesity = np.random.choice(["Yes", "No"], num_rows)
# metabolic_syndrome = np.random.choice(["Yes", "No"], num_rows)
# copd = np.random.choice(["Yes", "No"], num_rows)
# asthma = np.random.choice(["Yes", "No"], num_rows)
# depression = np.random.choice(["Yes", "No"], num_rows)
# anxiety = np.random.choice(["Yes", "No"], num_rows)
# stress_disorders = np.random.choice(["Yes", "No"], num_rows)
# ckd = np.random.choice(["Yes", "No"], num_rows)

# # Stroke risk as probability (0 to 1)
# stroke_risk = np.random.uniform(0, 1, num_rows)

# # Life Expectancy Calculation
# base_life_expectancy = 78  # Average life expectancy in developed countries

# # Applying adjustments based on health conditions
# life_expectancy = (
#     base_life_expectancy
#     - (np.where(heart_disease == "Yes", 7, 0))
#     - (np.where(hypertension == "Yes", 3, 0))
#     - (np.where(stroke == "Yes", 5, 0))
#     - (np.where(diabetes == "Yes", 4, 0))
#     - (np.where(obesity == "Yes", 4, 0))
#     - (np.where(metabolic_syndrome == "Yes", 3, 0))
#     - (np.where(copd == "Yes", 6, 0))
#     - (np.where(asthma == "Yes", 2, 0))
#     - (np.where(depression == "Yes", 3, 0))
#     - (np.where(anxiety == "Yes", 2, 0))
#     - (np.where(ckd == "Yes", 6, 0))
#     - (np.where(smoking_habits == "Daily", 5, 0))
#     - (np.where(drinking_habits == "Daily", 4, 0))
#     + (np.where(exercise_routine == "Heavy Activity", 3, 0))
#     + (np.where(diet_fruits_veggies == "Yes", 2, 0))
#     + (np.where(diet_water == "Yes", 1, 0))
# )

# # Ensuring life expectancy is never below the person's age
# life_expectancy = np.maximum(life_expectancy, age + 1)

# # Creating DataFrame
# df = pd.DataFrame({
#     "Heart Rate (BPM)": heart_rate,
#     "Blood Oxygen (SpO2)": blood_oxygen,
#     "Stress Level (HRV-Based)": stress_level_hrv,
#     "BMI": bmi,
#     "Age": age,
#     "Height (cm)": height,
#     "Weight (kg)": weight,
#     "Smoking Habits": smoking_habits,
#     "Drinking Habits": drinking_habits,
#     "Exercise Routine": exercise_routine,
#     "Medical History (Diabetes)": medical_diabetes,
#     "Medical History (Hypertension)": medical_hypertension,
#     "Medical History (Heart Disease)": medical_heart_disease,
#     "Medical History (Cancer)": medical_cancer,
#     "Dietary Habits (Fast Food)": diet_fast_food,
#     "Dietary Habits (Fruits/Vegetables)": diet_fruits_veggies,
#     "Dietary Habits (Caffeine)": diet_caffeine,
#     "Dietary Habits (Water)": diet_water,
#     "Dietary Habits (Red Meat)": diet_red_meat,
#     "Mental Health (Anxiety)": mental_health_anxiety,
#     "Mental Health (Depression)": mental_health_depression,
#     "Family History (Heart Disease)": family_heart_disease,
#     "Family History (Cancer)": family_cancer,
#     "Family History (Diabetes)": family_diabetes,
#     "Hydration (Glasses per Day)": glasses_per_day,
#     "Medication (Hypertension)": med_hypertension,
#     "Medication (Diabetes)": med_diabetes,
#     "Medication (Painkillers)": med_painkillers,
#     "Job Stress (Stress Level)": stress_level,
#     "Job Stress (Work Type)": work_type,
#     "Job Stress (Hours)": job_hours,
#     "Job Stress (Sleep Deprivation)": sleep_deprivation,
#     "Heart Disease": heart_disease,
#     "Hypertension": hypertension,
#     "Stroke": stroke,
#     "Type 2 Diabetes": diabetes,
#     "Obesity": obesity,
#     "Metabolic Syndrome": metabolic_syndrome,
#     "COPD": copd,
#     "Asthma": asthma,
#     "Depression": depression,
#     "Anxiety": anxiety,
#     "Stress Disorders": stress_disorders,
#     "Chronic Kidney Disease (CKD)": ckd,
#     "Stroke Risk": stroke_risk,
#     "Life Expectancy": life_expectancy
# })

# # Saving to CSV
# file_path = "health_dataset1_2M.csv"
# df.to_csv(file_path, index=False)







# # file_path
# import numpy as np
# import pandas as pd
# from scipy.special import expit  # Logistic function for probability calculations

# # Number of rows in dataset
# num_rows = 2_000_000

# # Base characteristics with realistic distributions
# age = np.clip(np.random.normal(45, 15, num_rows), 18, 90).astype(int)
# height = np.clip(np.random.normal(170, 10, num_rows), 120, 220).astype(int)  # Clipped height to realistic range
# weight = np.clip(np.random.normal(75, 15, num_rows), 40, 200).astype(int)  # Clipped weight to realistic range
# bmi = np.clip(weight / ((height / 100) ** 2), 10, 50)  # Clipped BMI to realistic range

# # Derived physiological features
# heart_rate = np.clip(np.random.normal(72, 10, num_rows), 50, 120).astype(int)
# blood_oxygen = np.clip(np.random.normal(97, 1.5, num_rows), 85, 100)
# stress_level_hrv = np.clip(
#     np.random.exponential(2, num_rows) + (bmi > 30) * 0.5 + (age > 60) * 0.3, 0.5, 10
# )

# # Function for age-related probabilities
# def age_related_probability(age, base_rate, age_factor):
#     return base_rate * (1 + age_factor * (age - 40) / 50)

# # Medical history probabilities based on age
# medical_diabetes = np.random.binomial(1, age_related_probability(age, 0.07, 0.8))
# medical_hypertension = np.random.binomial(1, age_related_probability(age, 0.15, 1.2))
# medical_heart_disease = np.random.binomial(1, age_related_probability(age, 0.05, 1.5))
# medical_cancer = np.random.binomial(1, 0.03 * (age / 80))

# # Lifestyle factors with correlations
# smoking_probs = np.array([0.6, 0.3, 0.1])  # Never, Occasionally, Daily
# smoking_habits = np.random.choice(["Never", "Occasionally", "Daily"], num_rows, p=smoking_probs)

# exercise_probs = np.array([0.3, 0.4, 0.2, 0.1])  # Sedentary, Light, Moderate, Heavy
# exercise_routine = np.random.choice([
#     "Sedentary", "Light Activity", "Moderate Activity", "Heavy Activity"
# ], num_rows, p=exercise_probs)

# # Function for calculating disease risk using logistic function
# def calculate_risk(factors, coefficients):
#     z = np.dot(factors, coefficients)
#     return expit(z)  # Sigmoid function for probability conversion

# # Heart Disease Risk Factors: Age, BMI, Smoking, Family History, Exercise
# heart_disease_risk = calculate_risk(
#     np.column_stack([
#         age / 80, bmi / 40, (smoking_habits == "Daily").astype(int), np.random.binomial(1, 0.2, num_rows), (exercise_routine == "Sedentary").astype(int)
#     ]),
#     [2.5, 1.8, 0.9, 0.7, 0.5]
# )
# heart_disease = np.random.binomial(1, heart_disease_risk)

# # Diabetes Risk Factors: BMI, Age, Family History, Exercise
# diabetes_risk = calculate_risk(
#     np.column_stack([
#         bmi / 40, age / 80, np.random.binomial(1, 0.15, num_rows), (exercise_routine == "Sedentary").astype(int)
#     ]),
#     [3.2, 1.5, 1.0, 0.6]
# )
# diabetes = np.random.binomial(1, diabetes_risk)

# # Hypertension Risk Factors: Age, BMI, Stress, Sodium Intake
# hypertension_risk = calculate_risk(
#     np.column_stack([
#         age / 80, bmi / 40, stress_level_hrv / 10, np.random.binomial(1, 0.4, num_rows)
#     ]),
#     [2.8, 2.2, 1.2, 0.8]
# )
# hypertension = np.random.binomial(1, hypertension_risk)

# # Life Expectancy Calculation (complex model)
# base_le = 79.0  # Base life expectancy
# le_adjustments = (
#     -5 * heart_disease 
#     -3 * diabetes 
#     -4 * hypertension 
#     -2 * (bmi > 30).astype(int) 
#     + 2 * (exercise_routine == "Heavy Activity").astype(int) 
#     -4 * (smoking_habits == "Daily").astype(int) 
#     + 3 * (exercise_routine == "Moderate Activity").astype(int) 
#     + 0.5 * (blood_oxygen > 95).astype(int) 
#     -0.1 * age 
#     + 0.05 * (100 - age)
# )

# life_expectancy = np.clip(base_le + le_adjustments + np.random.normal(0, 2, num_rows),
#                          age + 1, 100)

# # Creating DataFrame
# df = pd.DataFrame({
#     "Age": age,
#     "Height (cm)": height,
#     "Weight (kg)": weight,
#     "BMI": bmi,
#     "Heart Rate (BPM)": heart_rate,
#     "Blood Oxygen (SpO2)": blood_oxygen,
#     "Stress Level (HRV-Based)": stress_level_hrv,
#     "Smoking Habits": smoking_habits,
#     "Exercise Routine": exercise_routine,
#     "Medical History (Diabetes)": medical_diabetes,
#     "Medical History (Hypertension)": medical_hypertension,
#     "Medical History (Heart Disease)": medical_heart_disease,
#     "Medical History (Cancer)": medical_cancer,
#     "Heart Disease": heart_disease,
#     "Hypertension": hypertension,
#     "Type 2 Diabetes": diabetes,
#     "Life Expectancy": life_expectancy
# })

# # Save to CSV
# file_path = "health_dataset2_2M.csv"
# df.to_csv(file_path, index=False)
# print(f"Dataset saved as {file_path}")














# import numpy as np
# import pandas as pd

# # Set seed for reproducibility
# np.random.seed(42)

# # Number of samples
# n_samples = 2_000_000

# # Generate continuous features with realistic distributions
# age = np.random.randint(18, 90, size=n_samples)  # Age range from 18 to 90
# gender = np.random.choice(["Male", "Female"], size=n_samples)
# height = np.random.normal(170, 10, size=n_samples)  # Mean 170 cm, std 10
# weight = np.random.normal(70, 15, size=n_samples)  # Mean 70 kg, std 15
# bmi = weight / (height / 100) ** 2
# heart_rate = np.random.normal(75, 10, size=n_samples)  # BPM
# spo2 = np.clip(np.random.normal(97, 2, size=n_samples), 85, 100)  # Oxygen saturation
# stress_level = np.clip(np.random.normal(3, 1, size=n_samples), 1, 5)  # HRV-based stress level (1-5)
# exercise_routine = np.random.choice(["Sedentary", "Light Activity", "Moderate Activity", "Heavy Activity"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])

# # Categorical features with probabilities
# smoking_habits = np.random.choice(["Never", "Occasionally", "Daily"], size=n_samples, p=[0.7, 0.2, 0.1])
# drinking_habits = np.random.choice(["Never", "Occasionally", "Daily"], size=n_samples, p=[0.6, 0.3, 0.1])

# # Binary medical history
# has_diabetes = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
# has_hypertension = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
# has_heart_disease = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
# has_cancer = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])

# # Mental health & stress factors
# anxiety = np.random.randint(1, 6, size=n_samples)
# depression = np.random.randint(1, 6, size=n_samples)
# job_stress = np.random.randint(1, 6, size=n_samples)
# sleep_deprivation = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

# # Hydration & medication
# hydration = np.random.randint(2, 10, size=n_samples)  # Glasses of water
# med_hypertension = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
# med_diabetes = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# # Predict diseases
# heart_disease = (has_hypertension & has_diabetes & (age > 50)) | (np.random.rand(n_samples) < 0.05)
# hypertension = has_hypertension | (bmi > 30) | (np.random.rand(n_samples) < 0.1)
# stroke = (heart_disease & hypertension) | (np.random.rand(n_samples) < 0.02)
# type_2_diabetes = has_diabetes | ((bmi > 27) & (np.random.rand(n_samples) < 0.15))
# obesity = bmi > 30
# metabolic_syndrome = (obesity & has_hypertension & has_diabetes)
# copd = (smoking_habits == "Daily") | (np.random.rand(n_samples) < 0.03)
# asthma = (smoking_habits == "Daily") | (np.random.rand(n_samples) < 0.05)
# depression_disease = (depression >= 4) | (np.random.rand(n_samples) < 0.05)
# anxiety_disease = (anxiety >= 4) | (np.random.rand(n_samples) < 0.05)
# stress_disorders = (job_stress >= 4) | (np.random.rand(n_samples) < 0.05)
# ckd = has_diabetes & has_hypertension & (np.random.rand(n_samples) < 0.1)
# stroke_risk = np.clip(0.3 * heart_disease + 0.2 * hypertension + 0.15 * obesity + np.random.normal(0.2, 0.05, size=n_samples), 0, 1)

# # Calculate life expectancy using weighted features
# df["Life Expectancy"] = (80 - (df["Heart Disease"] * 8) - (df["Hypertension"] * 6) - 
#                          (df["Stroke"] * 10) - (df["Type 2 Diabetes"] * 6) - (df["Obesity"] * 5) - 
#                          (df["Metabolic Syndrome"] * 7) - (df["COPD"] * 9) - (df["Depression"] * 4) -
#                          (df["Chronic Kidney Disease"] * 8) - (df["Smoking"] == "Daily") * 5).clip(40, 100)

# # Create DataFrame
# df = pd.DataFrame({
#     "Age": age,
#     "Gender": gender,
#     "Height": height,
#     "Weight": weight,
#     "BMI": bmi,
#     "Heart Rate": heart_rate,
#     "SpO2": spo2,
#     "Stress Level": stress_level,
#     "Smoking Habits": smoking_habits,
#     "Drinking Habits": drinking_habits,
#     "Exercise Routine": exercise_routine,
#     "Has Diabetes": has_diabetes,
#     "Has Hypertension": has_hypertension,
#     "Has Heart Disease": has_heart_disease,
#     "Has Cancer": has_cancer,
#     "Anxiety": anxiety,
#     "Depression": depression,
#     "Job Stress": job_stress,
#     "Sleep Deprivation": sleep_deprivation,
#     "Hydration": hydration,
#     "Medication Hypertension": med_hypertension,
#     "Medication Diabetes": med_diabetes,
#     "Heart Disease": heart_disease,
#     "Hypertension": hypertension,
#     "Stroke": stroke,
#     "Type 2 Diabetes": type_2_diabetes,
#     "Obesity": obesity,
#     "Metabolic Syndrome": metabolic_syndrome,
#     "COPD": copd,
#     "Asthma": asthma,
#     "Depression Disease": depression_disease,
#     "Anxiety Disease": anxiety_disease,
#     "Stress Disorders": stress_disorders,
#     "Chronic Kidney Disease": ckd,
#     "Stroke Risk": stroke_risk,
#     "Life Expectancy": life_expectancy
# })

# # Save dataset to CSV
# df.to_csv("health_data3_2m.csv", index=False)

# print("Dataset generated and saved successfully!")










# import numpy as np
# import pandas as pd

# # Set seed for reproducibility
# np.random.seed(42)
# num_samples = 1_000_000

# # Generate continuous numerical data
# heart_rate = np.random.normal(loc=75, scale=12, size=num_samples).clip(50, 150)
# blood_oxygen = np.random.normal(loc=98, scale=1.5, size=num_samples).clip(85, 100)
# stress_level = np.random.normal(loc=50, scale=20, size=num_samples).clip(10, 100)
# bmi = np.random.normal(loc=25, scale=5, size=num_samples).clip(15, 40)
# age = np.random.randint(18, 90, num_samples)
# height = np.random.normal(loc=170, scale=10, size=num_samples).clip(140, 200)
# weight = bmi * ((height / 100) ** 2)

# # Generate categorical and binary data
# smoking = np.random.choice(["Never", "Occasionally", "Daily"], size=num_samples, p=[0.7, 0.2, 0.1])
# drinking = np.random.choice(["Never", "Occasionally", "Daily"], size=num_samples, p=[0.6, 0.3, 0.1])
# exercise = np.random.choice(["Sedentary", "Light Activity", "Moderate Activity", "Heavy Activity"], num_samples, p=[0.4, 0.3, 0.2, 0.1])
# medical_diabetes = np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
# medical_hypertension = np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
# medical_heart_disease = np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
# medical_cancer = np.random.choice([0, 1], num_samples, p=[0.98, 0.02])

# # Dietary habits
# fast_food = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
# fruits_vegetables = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
# caffeine = np.random.choice([0, 1], num_samples, p=[0.5, 0.5])
# water = np.random.choice([0, 1], num_samples, p=[0.3, 0.7])
# red_meat = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])

# # Mental health
# anxiety = np.random.randint(1, 6, num_samples)
# depression = np.random.randint(1, 6, num_samples)

# # Family history
# family_heart_disease = np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
# family_cancer = np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
# family_diabetes = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

# # Hydration & Medication
# hydration = np.random.randint(1, 15, num_samples)
# med_hypertension = np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
# med_diabetes = np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
# med_painkillers = np.random.choice([0, 1], num_samples, p=[0.85, 0.15])

# # Job & Lifestyle Stress
# stress_scale = np.random.randint(1, 6, num_samples)
# work_type = np.random.choice(["Desk Job", "Physical Work", "Field Work"], num_samples, p=[0.6, 0.3, 0.1])
# work_hours = np.random.randint(4, 14, num_samples)
# sleep_deprivation = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

# # Creating a DataFrame
# df = pd.DataFrame({
#     "Heart Rate": heart_rate, "Blood Oxygen": blood_oxygen, "Stress Level": stress_level, "BMI": bmi, "Age": age,
#     "Height": height, "Weight": weight, "Smoking": smoking, "Drinking": drinking, "Exercise": exercise,
#     "Diabetes": medical_diabetes, "Hypertension": medical_hypertension, "Heart Disease": medical_heart_disease, 
#     "Cancer": medical_cancer, "Fast Food": fast_food, "Fruits/Veg": fruits_vegetables, "Caffeine": caffeine, 
#     "Water": water, "Red Meat": red_meat, "Anxiety": anxiety, "Depression": depression, 
#     "Family Heart Disease": family_heart_disease, "Family Cancer": family_cancer, "Family Diabetes": family_diabetes, 
#     "Hydration": hydration, "Med Hypertension": med_hypertension, "Med Diabetes": med_diabetes, "Med Painkillers": med_painkillers, 
#     "Stress Scale": stress_scale, "Work Type": work_type, "Work Hours": work_hours, "Sleep Deprivation": sleep_deprivation
# })

# # Disease prediction (probabilistic based on risk factors)
# df["Heart Disease"] = ((df["Age"] > 50) & (df["Hypertension"] == 1) & (df["Family Heart Disease"] == 1)).astype(int)
# df["Stroke"] = ((df["Heart Disease"] == 1) & (df["Hypertension"] == 1)).astype(int)
# df["Type 2 Diabetes"] = ((df["BMI"] > 30) & (df["Family Diabetes"] == 1)).astype(int)
# df["Obesity"] = (df["BMI"] > 30).astype(int)
# df["Metabolic Syndrome"] = ((df["BMI"] > 27) & (df["Diabetes"] == 1) & (df["Hypertension"] == 1)).astype(int)
# df["COPD"] = ((df["Smoking"] == "Daily") & (df["Age"] > 50)).astype(int)
# df["Asthma"] = ((df["Family Heart Disease"] == 1) & (df["Exercise"] == "Sedentary")).astype(int)
# df["Depression"] = ((df["Stress Scale"] > 3) & (df["Anxiety"] > 3)).astype(int)
# df["Anxiety"] = (df["Anxiety"] > 3).astype(int)
# df["Stress Disorders"] = ((df["Stress Scale"] > 3) & (df["Sleep Deprivation"] == 1)).astype(int)
# df["Chronic Kidney Disease"] = ((df["Hypertension"] == 1) & (df["Diabetes"] == 1)).astype(int)

# # Stroke risk score (probabilistic scale 0-1)
# df["Stroke Risk"] = (df["Stroke"] * 0.8 + df["Hypertension"] * 0.3 + df["Heart Disease"] * 0.6).clip(0, 1)

# # Life Expectancy Calculation (base = 80, risk factor deductions)
# df["Life Expectancy"] = (80 - (df["Heart Disease"] * 8) - (df["Hypertension"] * 6) - 
#                          (df["Stroke"] * 10) - (df["Type 2 Diabetes"] * 6) - (df["Obesity"] * 5) - 
#                          (df["Metabolic Syndrome"] * 7) - (df["COPD"] * 9) - (df["Depression"] * 4) -
#                          (df["Chronic Kidney Disease"] * 8) - (df["Smoking"] == "Daily") * 5).clip(40, 100)

# # Save dataset
# file_path = "health_dataset4_2m.csv"
# df.to_csv(file_path, index=False)
# file_path

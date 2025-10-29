import pickle


datapoint = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}






with open('pipeline_v1.bin', 'rb') as f_in:
    model = pickle.load(f_in)





churn=model.predict_proba(datapoint)[0, 1]
print(churn)


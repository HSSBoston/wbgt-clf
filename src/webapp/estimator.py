import streamlit as st, pickle
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Hr, cloud cover, humidity, precip, temp, wind speed
# Alert level: 2
features = ["10", "3", "77", "0.0", "87", "4"]
# Alert level: 3
# features = ["14", "38", "73", "0.02", "89", "5"]

st.title("WBGT-based Heat Risk Estimator")
st.subheader(f"Current weather conditions:")

data = {
    "Time": [features[0] + ":00"],
    "Temperature (F)": [features[4]],
    "Humidity (%)": [features[2]],
    "Cloud cover (%)": [features[1]],
    "Wind speed (MPH)": [features[5]],
    "Precip (inch in 6 hrs)": [features[3]],
    }
df = pd.DataFrame(data)
st.dataframe(df.set_index(df.columns[0]))

clf = pickle.load(open("dt.pkl", "rb"))
# clf = joblib.load("dt.joblib")
# clf = joblib.load("dt.joblib")

y_predicted = clf.predict([features])
alertLevel = int(y_predicted[0])

if alertLevel == 3:
    st.subheader(f"Current Safety Alert Level: :red[{alertLevel} Extreme Condition]")
    st.write("No outdoor workouts/contests. Delay practice/competitions until a \
              cooler WBGT is reached.")
elif alertLevel == 2:
    st.subheader(f"Current Safety Alert Level: :red[{alertLevel} High Risk for Heat Related Illness]")
    st.write("Contests are permitted with additional hydration breaks. \
              Maximum outdoor practice time is 1 hour. No protective equipment \
              may be worn during practrice, and there may be no conditioning \
              activities. There must be 20-minute of rest breaks distributed \
              throughout the hour of practice.")
elif alertLevel == 1:
    st.subheader(f"Current Safety Alert Level: :orange[{alertLevel} Moderate Risk for Heat Related Illness]")
    st.write("Maximum outdoor practice time is 2 hours. Provide at least 4 separate \
              rest breaks each hour with a minimum duration of 4 minutes each.")
else:
    st.subheader(f"Current Safety Alert Level: :blue[{alertLevel} Good Condition]")
    st.write("Use discretion for intense or prolonged exercise. Provide at least \
              3 separate rest breaks each hour with a minimum duration of 4 \
              minutes each.")


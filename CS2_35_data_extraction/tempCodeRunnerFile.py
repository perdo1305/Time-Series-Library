# Create columns for cumulative charge and discharge capacities
            if i == 0:
                df["Cumulative_Charge_Capacity(Ah)"] = df["Charge_Capacity(Ah)"]
                df["Cumulative_Discharge_Capacity(Ah)"] = df["Discharge_Capacity(Ah)"]
import matplotlib.pyplot as plt
import numpy as np

def plot_model_actual_scat_comparision(model, scatter_index, actual_case_matrix: np.array((A,T))):
    plt.figure(figsize=(10, 5))

    plt.plot(model.index, model.values,
             label="Estimated number of cases", color="royalblue", linewidth=1)

    plt.scatter(
        scatter_index,
        #weekly_cases_filled[age_structured_data_mask].values,
        np.sum(actual_case_matrix, axis=0)[:len(model)],
        color="black",
        s=4,
        label="Observed data"
    )

    plt.title("Estimated and observed cases")
    plt.xlabel("Date")
    plt.ylabel("Weekly number of cases")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
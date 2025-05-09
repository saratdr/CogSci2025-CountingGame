from os.path import dirname, join
import matplotlib.pyplot as plt
import pandas as pd

CUR_DIR = dirname(__file__)
DATA_DIR = join(CUR_DIR, '../data')
PLOTS_DIR = join(CUR_DIR, 'plots')

df = pd.read_csv(join(DATA_DIR, 'full_data.csv'))

# Creates a plot depicting average accuracy and response times for each forgetting operation and round.
# The plot is saved as a PDF file in the analysis/plots directory.
def generalDataAnalysis():
    agg_data = df.groupby("condition").agg(
        Accuracy=("correct", "mean"),  
        Response_Time=("rt", "median"), 
    ).reset_index()

    agg_data["Forgetting_Type"] = agg_data["condition"].str.extract(r"(.+)_r")[0].str.capitalize()
    agg_data["Forgetting_Type"] = agg_data["Forgetting_Type"].replace({"Fadingout": "Fading out"})
    agg_data["Round"] = agg_data["condition"].str.extract(r"_r(\d)")[0]

    base_color = "skyblue"
    lighter_color = "lightblue"
    agg_data["Color"] = agg_data["Round"].map({"1": base_color, "2": lighter_color})

    forgetting_types = agg_data["Forgetting_Type"].unique()
    bar_positions = []
    current_pos = 0
    bar_width = 0.4

    for ftype in forgetting_types:
        positions = [current_pos, current_pos + bar_width]
        bar_positions.extend(positions)
        current_pos += 1

    agg_data["Bar_Position"] = bar_positions

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars = ax1.bar(
        agg_data["Bar_Position"], 
        agg_data["Accuracy"] * 100,
        color=agg_data["Color"], 
        width=bar_width, 
        label="Mean Accuracy"
    )

    for bar, accuracy in zip(bars, agg_data["Accuracy"] * 100):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{accuracy:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=10
        )

    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 100)

    xticks_positions = [
        (bar_positions[i] + bar_positions[i + 1]) / 2 for i in range(0, len(bar_positions), 2)
    ]
    xticks_labels_top = ["Round 1", "Round 2"] * len(forgetting_types)
    xticks_labels_bottom = forgetting_types

    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(xticks_labels_top, fontsize=10)

    for i, txt in enumerate(xticks_labels_bottom):
        ax1.text(
            xticks_positions[i], 
            -0.1, 
            txt, 
            ha="center", 
            va="top", 
            fontsize=10, 
            transform=ax1.get_xaxis_transform()
        )

    ax2 = ax1.twinx()
    response_times = agg_data["Response_Time"] / 1000
    line = ax2.plot(
        agg_data["Bar_Position"],  
        response_times,
        color="orange",
        marker="o",
        markersize=10, 
        label="Median Response Time",
        linewidth=2
    )

    for pos, value in zip(agg_data["Bar_Position"], response_times):
        ax2.text(
            pos, 
            value + 0.2, 
            f"{value:.2f}s", 
            ha="center", 
            va="top", 
            fontsize=10, 
            color="black"
        )

    ax2.set_ylabel("Response Time (seconds)", fontsize=12)
    ax2.set_ylim(0, (agg_data["Response_Time"].max() / 1000) * 1.2)

    ax1.legend(loc="upper left", bbox_to_anchor=(0.1, 1))
    ax2.legend(loc="upper right", bbox_to_anchor=(0.9, 1))

    plt.title("Accuracy and Response Times by Forgetting Type and Round", fontsize=14)
    plt.tight_layout()

    plt.savefig(join(PLOTS_DIR, "general_accuracy_rt.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


generalDataAnalysis()
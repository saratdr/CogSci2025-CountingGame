import pandas as pd
from os.path import dirname, join
from tabulate import tabulate
from scipy.stats import spearmanr, pointbiserialr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
pd.options.mode.chained_assignment = None

# Directory paths
CUR_DIR = dirname(__file__)
DATA_DIR = join(CUR_DIR, '../data')

# Read participant data and trial info data
data = pd.read_csv(join(DATA_DIR, 'full_data.csv'))
trial_info = pd.read_csv(join(DATA_DIR, 'trial_info.csv'))

# Remove outliers based on participant accuracy
# Outliers = participants whose accuracy is more than 2 standard deviations away from the mean accuracy
def removeOutliers(df):
    original_len = len(df["id"].unique())
    participant_accuracy = df.groupby('id')['correct'].mean()
    mean_accuracy = participant_accuracy.mean()
    std_accuracy = participant_accuracy.std()
    outliers = participant_accuracy[(participant_accuracy < mean_accuracy - 2 * std_accuracy) | (participant_accuracy > mean_accuracy + 2 * std_accuracy)]
    after_len = len(df[~df['id'].isin(outliers.index)]["id"].unique())
    print(f"Original length: {original_len}, After removing outliers: {after_len}\n")
    return df[~df['id'].isin(outliers.index)]

# Get overall information about the data
# Mean, median and standard deviation of correctness and response time
def getOverallInfo(df):
    accuracy_mean = df["correct"].mean() * 100
    accuracy_median = df["correct"].median() * 100
    accuracy_std = df["correct"].std() * 100

    rt_median = df["rt"].median() / 1000
    rt_mean = df["rt"].mean() / 1000
    rt_std = df["rt"].std() / 1000
    
    print("Correctness: Mean =", f"{accuracy_mean:.2f}%, Median =", f"{accuracy_median:.2f}%, SD =", f"{accuracy_std:.2f}%")
    print("Response Time: Mean =", f"{rt_mean:.2f}s, Median =", f"{rt_median:.2f}s, SD =", f"{rt_std:.2f}s")

    noncrit_trials = trial_info[trial_info["crit"] == "nc"]
    crit_trials = trial_info[trial_info["crit"] != "nc"]

    noncrit_accuracy = df.loc[df["trial_id"].isin(noncrit_trials["trial_id"])]["correct"].mean() * 100
    crit_accuracy = df.loc[df["trial_id"].isin(crit_trials["trial_id"])]["correct"].mean() * 100
    print("Non-critical Trials Accuracy: ", f"{noncrit_accuracy:.2f}%")
    print("Critical Trials Accuracy: ", f"{crit_accuracy:.2f}%")
    noncrit_rt = df.loc[df["trial_id"].isin(noncrit_trials["trial_id"])]["rt"].median() / 1000
    crit_rt = df.loc[df["trial_id"].isin(crit_trials["trial_id"])]["rt"].median() / 1000
    print("Non-critical Trials RT: ", f"{noncrit_rt:.2f}s")
    print("Critical Trials RT: ", f"{crit_rt:.2f}s")
    print()

# Get correctness and response time for each round
# Mean accuracy and median response time for each round
def getRoundCorrectness(df):
    table = []
    for_plot = {"forgetting_types": [], "accuracy": [], "response_time": []}
    for ft, group in df.groupby('condition'):
        mean_accuracy = group["correct"].mean() * 100
        median_rt = group["rt"].median() / 1000
        table.append([ft.capitalize(), f"{mean_accuracy:.2f}%", f"{median_rt:.2f}s"])
        for_plot["forgetting_types"].append(ft.capitalize())
        for_plot["accuracy"].append(float(f"{mean_accuracy:.2f}"))
        for_plot["response_time"].append(float(f"{median_rt:.2f}"))

    print(tabulate(table, headers=["Round", "Accuracy (Mean)", "RT (Median)"]))
    print("\n")

# Wrapper for overall general analysis
def aggregateAnalysis():
    print("--- Aggregate Analysis ---")
    getOverallInfo(data)
    getRoundCorrectness(data)

# Statistics function to calculate different statistics
# Spearman correlation, Point Biserial correlation and Mann Whitney U test
def doStatistics(data, method, label, to_print=False):
    if to_print:
        print(f"{label} - {method.capitalize()}")
    if method == "spearman":
        stat, p = spearmanr(data[0], data[1])
        if to_print:
            print(f"r = {stat:.2f}, p = {p:.3f}")
    elif method == "pointbiserial":
        stat, p = pointbiserialr(data[0], data[1])
        if to_print:
            print(f"r = {stat:.2f}, p = {p:.3f}")
    elif method == "mannwhitneyu":
        stat, p = mannwhitneyu(data[0], data[1])
        if to_print:
            print(f"U = {stat:.2f}, p = {p:.3f}")
    else:
        print("Not a valid method")
    return stat, p

# Clean print function for statistics results
def printStatResults(p_value, corrected_p_value, stat, method, label):
    print(f"{label} - {method.capitalize()}")
    print(f"stat = {stat:.2f}, p = {p_value:.3f}, corrected p = {corrected_p_value:.3f}\n")

# Analyze trial color distribution features
# "windom" - Winner Dominance - the ratio between the winner and all elements, indicating if a winner is clearly dominant with a higher value.
# "lead" - Lead - the difference between the winner and the second-place color, a smaller value indicating “tight” competition. 
# "bal" - Imbalance - quantifies how evenly distributed the three eligible colors are by considering the difference between the highest and lowest counts with a lower value suggesting a more balanced trial.
# "windom_lead" - WinDom x Lead - the interaction between winner dominance and lead - To analyze the effect of clearly dominant winners with no close competition
def trialDistributionFeatures():
    print("\n--- Trial Distribution Features ---\n")
    phase0 = data[data["rule_id_exp"] == 0]
    metrics = ["windom", "lead", "imbal"]
    phase0["lead"] = phase0["lead"].apply(lambda x: x/16)
    phase0["windom_lead"] = phase0["windom"] * phase0["lead"]
    metrics.append("windom_lead")
    stats = []
    p_values_to_correct = []
    
    for metric in metrics:
        stat, p = doStatistics([phase0["correct"], phase0[metric]], "pointbiserial", "Correctness vs. " + metric)
        stats.append(stat)
        p_values_to_correct.append(p)
        stat, p = doStatistics([phase0["rt"], phase0[metric]], "spearman", "RT vs. " + metric)
        stats.append(stat)
        p_values_to_correct.append(p)
    _, corrected_pvals, _, _ = multipletests(p_values_to_correct, alpha=0.05, method='holm')

    for idx in range(len(stats)):
        printStatResults(p_values_to_correct[idx], corrected_pvals[idx], stats[idx], "pointbiserial" if idx % 2 == 0 else "spearman", "Correctness vs. " + metrics[idx // 2] if idx % 2 == 0 else "RT vs. " + metrics[idx // 2])

# Rule Operations
# The effect of count-twice and count-as rules on correctness and response time
def ruleOperations():
    print("\n--- Rule Operations ---\n")
    df_filtered = data[data["rule_id_exp"] == 1].copy()
    df_filtered["trial_type"] = df_filtered["crit"].apply(lambda x: "critical" if x in ["r1", "r1r2", "r1r2r3"] else "non-critical")
    df_filtered.loc[df_filtered["rule_type"] == "count_gray", "rule_type"] = "count_as"
    df_filtered.loc[df_filtered["rule_id_exp"] == 0, "rule_type"] = "none"

    df_critical = df_filtered[df_filtered["trial_type"] == "critical"]

    df_ctwice = df_filtered[df_filtered["rule_type"] == "count_twice"]
    df_cas = df_filtered[df_filtered["rule_type"] == "count_as"]

    print("Count Twice vs. Count As")
    model_ctwice_vs_cas_crit = smf.glm("correct ~ rule_type", data=df_critical, family=sm.families.Binomial()).fit()
    print(model_ctwice_vs_cas_crit.summary())

    print("Count Twice: Critical vs. Non-Critical")
    model_ctwice_crit_vs_noncrit = smf.glm("correct ~ trial_type", data=df_ctwice, family=sm.families.Binomial()).fit()
    print(model_ctwice_crit_vs_noncrit.summary())

    print("Count As: Critical vs. Non-Critical")
    model_cas_crit_vs_noncrit = smf.glm("correct ~ trial_type", data=df_cas, family=sm.families.Binomial()).fit()
    print(model_cas_crit_vs_noncrit.summary())

    p_values = [
        model_ctwice_vs_cas_crit.pvalues["rule_type[T.count_twice]"],
        model_ctwice_crit_vs_noncrit.pvalues["trial_type[T.non-critical]"], 
        model_cas_crit_vs_noncrit.pvalues["trial_type[T.non-critical]"]
    ]
    corrected_p_values = multipletests(p_values, method="holm")[1]
    print(f"Corrected p-values: {corrected_p_values}\n")

    rt_ctwice_crit = df_ctwice[df_ctwice["trial_type"] == "critical"]["rt"]
    rt_cas_crit = df_cas[df_cas["trial_type"] == "critical"]["rt"]
    stat_rt_ctwice_cas, p_rt_ctwice_cas = doStatistics([rt_ctwice_crit, rt_cas_crit], "mannwhitneyu", "RT Count Twice vs. Count As")

    rt_ctwice_noncrit = df_ctwice[df_ctwice["trial_type"] == "non-critical"]["rt"]
    stat_rt_ctwice, p_rt_ctwice = doStatistics([rt_ctwice_crit, rt_ctwice_noncrit], "mannwhitneyu", "RT Count Twice: Critical vs. Non-Critical")

    rt_cas_noncrit = df_cas[df_cas["trial_type"] == "non-critical"]["rt"]
    stat_rt_cas, p_rt_cas = doStatistics([rt_cas_crit, rt_cas_noncrit], "mannwhitneyu", "RT Count As: Critical vs. Non-Critical")

    corrected_p_values_rt = multipletests([p_rt_ctwice_cas, p_rt_ctwice, p_rt_cas], method="holm")[1]

    printStatResults(p_rt_ctwice_cas, corrected_p_values_rt[0], stat_rt_ctwice_cas, "mannwhitneyu", "RT Count Twice vs. Count As")
    printStatResults(p_rt_ctwice, corrected_p_values_rt[1], stat_rt_ctwice, "mannwhitneyu", "RT Count Twice: Critical vs. Non-Critical")
    printStatResults(p_rt_cas, corrected_p_values_rt[2], stat_rt_cas, "mannwhitneyu", "RT Count As: Critical vs. Non-Critical")

# Rule Interactions
# The effect of multiple rules being active at the same time on correctness and response time
def ruleInteractions():
    print("\n--- Rule Interactions ---\n")

    df_filtered = data[
    (((data["condition"] == "contraction_r1") & (data["rule_id_exp"] == 2)) |
    ((data["condition"] == "contraction_r2") & (data["rule_id_exp"] == 3)))].copy()

    df_filtered["trial_type"] = df_filtered["crit"].apply(
        lambda x: "critical_r2" if x == "r2" 
        else "critical_r1r2" if x == "r1r2" 
        else "non_critical")

    model_crit_r2 = smf.glm("correct ~ condition", data=df_filtered[df_filtered["trial_type"] == "critical_r2"], family=sm.families.Binomial()).fit()
    print("\nCritical R2")
    print(model_crit_r2.summary())

    model_crit_r1r2 = smf.glm("correct ~ condition", data=df_filtered[df_filtered["trial_type"] == "critical_r1r2"], family=sm.families.Binomial()).fit()
    print("\nCritical R1R2")
    print(model_crit_r1r2.summary())

    model_noncrit = smf.glm("correct ~ condition", data=df_filtered[df_filtered["trial_type"] == "non_critical"], family=sm.families.Binomial()).fit()
    print("\nNon-Critical")
    print(model_noncrit.summary())

    p_values = [
        model_crit_r2.pvalues["condition[T.contraction_r2]"],
        model_crit_r1r2.pvalues["condition[T.contraction_r2]"],
        model_noncrit.pvalues["condition[T.contraction_r2]"]
    ]

    corrected_p_values = multipletests(p_values, method="holm")[1]
    print(f"Corrected p-values: {corrected_p_values}\n")

    rt_r2_round1 = df_filtered[(df_filtered["trial_type"] == "critical_r2") & (df_filtered["condition"] == "contraction_r1")]["rt"]
    rt_r2_round2 = df_filtered[(df_filtered["trial_type"] == "critical_r2") & (df_filtered["condition"] == "contraction_r2")]["rt"]
    stat_rt_r2, p_rt_r2 = doStatistics([rt_r2_round1, rt_r2_round2], "mannwhitneyu", "RT Critical R2: Round 1 vs. Round 2")

    rt_r1r2_round1 = df_filtered[(df_filtered["trial_type"] == "critical_r1r2") & (df_filtered["condition"] == "contraction_r1")]["rt"]
    rt_r1r2_round2 = df_filtered[(df_filtered["trial_type"] == "critical_r1r2") & (df_filtered["condition"] == "contraction_r2")]["rt"]
    stat_rt_r1r2, p_rt_r1r2 = doStatistics([rt_r1r2_round1, rt_r1r2_round2], "mannwhitneyu", "RT Critical R1R2: Round 1 vs. Round 2")

    rt_nc_round1 = df_filtered[(df_filtered["trial_type"] == "non_critical") & (df_filtered["condition"] == "contraction_r1")]["rt"]
    rt_nc_round2 = df_filtered[(df_filtered["trial_type"] == "non_critical") & (df_filtered["condition"] == "contraction_r2")]["rt"]
    stat_rt_nc, p_rt_nc = doStatistics([rt_nc_round1, rt_nc_round2], "mannwhitneyu", "RT Non-Critical: Round 1 vs. Round 2")

    corrected_p_values_rt = multipletests([p_rt_r2, p_rt_r1r2, p_rt_nc], method="holm")[1]

    printStatResults(p_rt_r2, corrected_p_values_rt[0], stat_rt_r2, "mannwhitneyu", "RT Critical R2: Round 1 vs. Round 2")
    printStatResults(p_rt_r1r2, corrected_p_values_rt[1], stat_rt_r1r2, "mannwhitneyu", "RT Critical R1R2: Round 1 vs. Round 2")
    printStatResults(p_rt_nc, corrected_p_values_rt[2], stat_rt_nc, "mannwhitneyu", "RT Non-Critical: Round 1 vs. Round 2")

# Rule Interaction and Operations
# Difference between rule types when other rules are simultaneously active
def ruleInteractionsOperations():
    print("\n--- Rule Interactions Operations ---\n")
    df_filtered = data[
        (data["condition"].isin(["fadingout_r1", "fadingout_r2"])) & (data["rule_id_exp"] == 3)].copy()

    df_filtered["crit"] = df_filtered["crit"].apply(lambda x: "critical" if x == "r3" else "non_critical")

    model_crit = smf.glm("correct ~ condition", data=df_filtered[df_filtered["crit"] == "critical"], family=sm.families.Binomial()).fit()
    print("\nCritical R3")
    print(model_crit.summary())

    rt_ctwice_crit = df_filtered[(df_filtered["crit"] == "critical") & (df_filtered["condition"] == "fadingout_r1")]["rt"]
    rt_cas_crit = df_filtered[(df_filtered["crit"] == "critical") & (df_filtered["condition"] == "fadingout_r2")]["rt"]

    stat_rt_crit, p_rt_crit = doStatistics([rt_ctwice_crit, rt_cas_crit], "mannwhitneyu", "RT Critical: Count Twice vs. Count As")
    printStatResults(p_rt_crit, p_rt_crit, stat_rt_crit, "mannwhitneyu", "RT Critical: Count Twice vs. Count As")

# Temporal Effects
# How temporal separation between a rule and its revision affects task difficulty.
def temporalEffects():
    print("\n--- Temporal Effects ---\n")
    df_filtered = data[
        ((data["condition"] == "revision_r1") & (data["rule_id_exp"] == 2)) |
        ((data["condition"] == "revision_r2") & (data["rule_id_exp"] == 3))].copy()

    df_filtered = df_filtered[df_filtered["crit"] == "r1r2"]

    model_crit_r1r2 = smf.glm("correct ~ condition", data=df_filtered, family=sm.families.Binomial()).fit()
    print("\nCritical R1R2")
    print(model_crit_r1r2.summary())

    rt_r1 = df_filtered[df_filtered["condition"] == "revision_r1"]["rt"]
    rt_r2 = df_filtered[df_filtered["condition"] == "revision_r2"]["rt"]
    stat_rt, p_rt = doStatistics([rt_r1, rt_r2], "mannwhitneyu", "RT Critical R1R2: Round 1 vs. Round 2")
    printStatResults(p_rt, p_rt, stat_rt, "mannwhitneyu", "RT Critical R1R2: Round 1 vs. Round 2")

# Scope of Rules
# Whether applying rules to a more specific subset of shapes (e.g., “Gray triangles count as red”) is more difficult than applying them to a broader category (e.g., “Gray shapes count as red”)
def scopeOfRules():
    print("\n--- Scope of Rules ---\n")
    df_filtered = data[
        (data["condition"].isin(["conditionalization_r1", "conditionalization_r2"])) & (data["rule_id_exp"] == 1)].copy()

    df_filtered["crit"] = df_filtered["crit"].apply(lambda x: "critical" if x == "r1" else "non_critical")

    model_crit = smf.glm("correct ~ condition", data=df_filtered[df_filtered["crit"] == "critical"], family=sm.families.Binomial()).fit()
    print("\nCritical R1")
    print(model_crit.summary())

    rt_r1 = df_filtered[(df_filtered["crit"] == "critical") & (df_filtered["condition"] == "conditionalization_r1")]["rt"]
    rt_r2 = df_filtered[(df_filtered["crit"] == "critical") & (df_filtered["condition"] == "conditionalization_r2")]["rt"]

    stat_rt, p_rt = doStatistics([rt_r1, rt_r2], "mannwhitneyu", "RT Critical: Round 1 vs. Round 2")
    printStatResults(p_rt, p_rt, stat_rt, "mannwhitneyu", "RT Critical: Round 1 vs. Round 2")

aggregateAnalysis()
data = removeOutliers(data)
aggregateAnalysis()
trialDistributionFeatures()
ruleOperations()
ruleInteractions()
ruleInteractionsOperations()
temporalEffects()
scopeOfRules()

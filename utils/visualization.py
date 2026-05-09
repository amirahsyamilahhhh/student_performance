"""Visualization utilities for EDA and model output."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURE_DIR


def _save_or_show(filename: str | None = None):
    """Save the current plot when a filename is provided; otherwise show it."""
    plt.tight_layout()
    if filename:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(FIGURE_DIR) / filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_age_performance(df, filename: str | None = "age_performance.png"):
    """Plot average attendance percentage and overall score by age."""
    grouped_df = df.groupby("age")[["attendance_percentage", "overall_score"]].mean()
    grouped_df.plot(kind="bar", figsize=(9, 5), title="Average Attendance and Overall Score by Age")
    plt.xlabel("Age")
    plt.ylabel("Average Value")
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    _save_or_show(filename)


def plot_gender_school_performance(df, filename: str | None = "gender_school_performance.png"):
    """Plot average overall score by gender and school type."""
    pivot_df = df.pivot_table(values="overall_score", index="gender", columns="school_type", aggfunc="mean")
    pivot_df.plot(kind="bar", figsize=(9, 5), title="Average Overall Score by Gender and School Type")
    plt.xlabel("Gender")
    plt.ylabel("Average Overall Score")
    plt.xticks(rotation=0)
    plt.legend(title="School Type")
    _save_or_show(filename)


def plot_parent_education(df, filename: str | None = "parent_education.png"):
    """Plot average overall score by parent education level."""
    parent_education = df.groupby("parent_education")["overall_score"].mean().sort_values(ascending=False)
    parent_education.plot(kind="bar", figsize=(8, 5), title="Average Overall Score by Parent Education")
    plt.xlabel("Parent Education")
    plt.ylabel("Overall Score")
    plt.xticks(rotation=45)
    _save_or_show(filename)


def plot_internet_activity_heatmap(df, filename: str | None = "internet_activity_heatmap.png"):
    """Plot heatmap of performance by internet access and extra activities."""
    pivot_df = df.pivot_table(values="overall_score", index="internet_access", columns="extra_activities", aggfunc="mean")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="Oranges", linewidths=0.5)
    plt.title("Average Overall Score by Internet Access and Extra Activities")
    plt.xlabel("Extra Activities")
    plt.ylabel("Internet Access")
    _save_or_show(filename)


def plot_study_hours_regression(df, filename: str | None = "study_hours_regression.png"):
    """Plot relationship between study hours and overall score."""
    plt.figure(figsize=(8, 5))
    sns.regplot(x="study_hours", y="overall_score", data=df)
    plt.title("Overall Score vs Study Hours")
    plt.xlabel("Study Hours")
    plt.ylabel("Overall Score")
    plt.grid(True, linestyle="--", alpha=0.7)
    _save_or_show(filename)


def plot_final_grade_distribution(df, filename: str | None = "final_grade_distribution.png"):
    """Plot final grade distribution as a pie chart."""
    grade_counts = df["final_grade"].value_counts().sort_index()
    plt.figure(figsize=(6, 6))
    plt.pie(grade_counts, labels=grade_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Final Grade Distribution")
    _save_or_show(filename)


def generate_all_plots(df):
    """Generate and save all visualization outputs."""
    plot_age_performance(df)
    plot_gender_school_performance(df)
    plot_parent_education(df)
    plot_internet_activity_heatmap(df)
    plot_study_hours_regression(df)
    plot_final_grade_distribution(df)

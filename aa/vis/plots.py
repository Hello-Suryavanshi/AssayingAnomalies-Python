import pandas as pd
import matplotlib.pyplot as plt


def line_cumret(
    df: pd.DataFrame, date_col: str, ret_col: str, title: str = "Cumulative Return"
):
    s = (1.0 + df[ret_col].fillna(0.0)).cumprod()
    plt.figure()
    plt.plot(df[date_col], s)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.tight_layout()

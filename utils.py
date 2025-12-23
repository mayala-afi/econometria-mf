import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd


def plot_resid(adj_vals, resids):
    plt.figure(figsize=(8, 6))
    plt.scatter(adj_vals, resids, alpha=0.6, color="#37BBF4")
    plt.xlabel("Valores Ajustados")
    plt.ylabel("Residuos")
    plt.title("Residuos vs Valores Ajustados")
    plt.axhline(0, color="#062D3F", alpha=0.7, linestyle="--")
    plt.grid(True, alpha=0.3)
    plt.show()


def predict_plot(df, title):
    if hasattr(df.index, "to_timestamp"):
        x = df.index.to_timestamp()
    else:
        x = df.index

    y_real = df.iloc[:, 0]
    y_hat = df.iloc[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, y_real, color="#062D3F", label="Observado", linewidth=2)
    ax.plot(x, y_hat, color="#37BBF4", label="Predicho", linewidth=2, linestyle="--")

    ax.set_title(title)

    ax.yaxis.grid(True, linestyle="-", alpha=0.3)
    ax.xaxis.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc="upper left", ncol=2, frameon=False)

    plt.show()


def plot_single_line(df, title, y_label):
    if hasattr(df.index, "to_timestamp"):
        x = df.index.to_timestamp()
    else:
        x = df.index

    y = df.iloc[:, 0]

    plt.figure(figsize=(10, 5))

    plt.plot(x, y, color="#062D3F", label=y_label, linewidth=2)

    plt.title(title)

    plt.gca().yaxis.grid(True, linestyle="-", alpha=0.3)
    plt.gca().xaxis.grid(False)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.legend(loc="upper left", ncol=2, frameon=False)

    plt.show()


def plot_acf_pacf(df, var="infl_m", lags=24, title_suffix=""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(df[var], lags=lags, ax=ax[0])
    ax[0].set_title(f"ACF {title_suffix}")
    plot_pacf(df[var], lags=lags, ax=ax[1])
    ax[1].set_title(f"PACF {title_suffix}")

    plt.show()


def predict_plot_multi(df, title, labels=None, df_train=None):
    colors = ["#062D3F", "#37BBF4", "#C8DAE2"]
    if hasattr(df.index, "to_timestamp"):
        x = df.index.to_timestamp()
    else:
        x = df.index

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(df.shape[1]):
        ax.plot(
            x,
            df.iloc[:, i],
            color=colors[i % len(colors)],
            linewidth=2,
            linestyle="-" if i == 0 else "--",
            label=labels[i] if labels else f"Serie {i}",
        )

    ax.set_title(title)

    ax.yaxis.grid(True, linestyle="-", alpha=0.3)
    ax.xaxis.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc="upper left", ncol=df.shape[1], frameon=False)
    if df_train is not None:
        plt.axvline(df_train.index[-1], color="gray", linestyle=":", alpha=0.7)

    plt.show()


def adf_test(series, name):
    res = adfuller(series)
    print(f"{name}:\n -ADF={res[0]:.3f}\n -p-value={res[1]:.3f}\n")


def clean_fred_df(df, data_colname, data_newname, datecol="observation_date"):
    df.rename(columns={datecol: "fecha", data_colname: data_newname}, inplace=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df.set_index("fecha", inplace=True)
    return df


def irf_plot(impulse, response, irf):
    import matplotlib.pyplot as plt

    irf.plot(impulse=impulse, response=response)
    plt.show()


def plot_forecast(df_obs, df_forecast, var_name, title):
    import numpy as np

    df_plot = df_obs[[var_name]].copy()
    df_plot.columns = ["obs"]
    df_plot["yhat"] = np.nan
    df_plot = pd.concat(
        [df_plot, df_forecast[[var_name]].rename(columns={var_name: "yhat"})]
    )
    predict_plot(df_plot, title=title)

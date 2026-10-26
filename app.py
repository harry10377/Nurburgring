import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from pathlib import Path

from dash import Dash, html, dcc, Input, Output, State

app = Dash(__name__)

server = app.server

# --------------------------------------------------
# COLOUR PALETTE (Nordschleife: forest, kerbs, asphalt)
# --------------------------------------------------

ASPHALT = "#1F2A30"   # header, headings, text
FOREST = "#2F6B4F"    # fast / good
AMBER = "#E2A93B"     # mid
KERB_RED = "#C8372D"  # slow / accent
TARMAC = "#E6EAE8"    # page background
STEEL = "#4A7C8C"     # secondary accent

# Fast (green) -> slow (red)
SPEED_SCALE = [
    [0.0, FOREST],
    [0.5, AMBER],
    [1.0, KERB_RED],
]

# Distinct colours for the spider chart traces
TRACE_COLORS = [FOREST, KERB_RED, STEEL, AMBER, "#7B4B94"]

STAT_ACCENTS = [FOREST, KERB_RED, STEEL, AMBER]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

APP_DIR = Path(__file__).parent


def data_file(name):
    """Find a data file in data/ (the submission structure) or next to app.py."""
    for folder in (APP_DIR / "data", APP_DIR):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(
        f"Could not find {name}. Put it in a folder called 'data' next to app.py."
    )


data = pd.read_csv(data_file("laptimes.csv"))


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def time_to_seconds(t):
    """Convert M:SS.ss lap times to seconds."""
    if pd.isna(t):
        return None

    match = re.fullmatch(
        r"(\d+):(\d{1,2}(?:\.\d+)?)",
        str(t).strip()
    )

    if not match:
        return None

    minutes = int(match.group(1))
    seconds = float(match.group(2))

    if seconds >= 60:
        return None

    return minutes * 60 + seconds


def format_lap_time(seconds):
    """Format seconds as M:SS.sss."""
    if pd.isna(seconds):
        return "N/A"

    minutes = int(seconds // 60)
    remaining = seconds % 60

    return f"{minutes}:{remaining:06.3f}"


# --------------------------------------------------
# CLEAN LAPTIMES DATA
# --------------------------------------------------

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Support common alternative column names
aliases = {
    "vehicle": ["vehicle", "car", "model", "name"],
    "driver": ["driver", "pilot"],
    "time_seconds": ["time_seconds", "lap_time_seconds"],
    "time_text": ["time_text", "time", "lap_time"],
    "rank": ["rank", "position"],
    "power_ps": ["power_ps", "power", "ps", "horsepower"],
    "weight_kg": ["weight_kg", "weight", "weight (kg)"],
    "power_to_weight": [
        "power_to_weight", "power/weight", "power_to_weight_ratio"
    ],
}

for target, options in aliases.items():
    if target not in data.columns:
        for option in options:
            if option in data.columns:
                data = data.rename(columns={option: target})
                break

# Require vehicle and lap time
if "vehicle" not in data.columns:
    raise ValueError(
        "laptimes.csv needs a vehicle, car, model, or name column."
    )

if "time_seconds" not in data.columns:
    if "time_text" in data.columns:
        data["time_seconds"] = data["time_text"].apply(
            time_to_seconds
        )
    else:
        raise ValueError(
            "laptimes.csv needs time_seconds or a lap-time column "
            "formatted as M:SS.ss."
        )

data["time_seconds"] = pd.to_numeric(
    data["time_seconds"], errors="coerce"
)

if "time_text" not in data.columns:
    data["time_text"] = data["time_seconds"].apply(
        format_lap_time
    )

if "driver" not in data.columns:
    data["driver"] = "Unknown"
else:
    data["driver"] = data["driver"].fillna("Unknown").astype(str)

if "rank" not in data.columns:
    data["rank"] = (
        data["time_seconds"].rank(method="first").astype("Int64")
    )
else:
    data["rank"] = pd.to_numeric(
        data["rank"], errors="coerce"
    )

# Add missing performance columns if needed
for col in ["power_ps", "weight_kg", "power_to_weight"]:
    if col not in data.columns:
        data[col] = float("nan")
    else:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Calculate ratio where possible
missing_ratio = (
    data["power_to_weight"].isna()
    & data["power_ps"].notna()
    & data["weight_kg"].gt(0)
)

data.loc[missing_ratio, "power_to_weight"] = (
    data.loc[missing_ratio, "power_ps"]
    / data.loc[missing_ratio, "weight_kg"]
)

data = data.dropna(
    subset=["vehicle", "time_seconds"]
).copy()

# Remove stray spaces around vehicle names
data["vehicle"] = data["vehicle"].astype(str).str.strip()


# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

def summary_stats(df):
    """Return (fastest_time, fastest_display, fastest_vehicle, median)."""
    valid = df.dropna(subset=["time_seconds"])

    if valid.empty:
        return "N/A", "N/A", "N/A", "N/A"

    fastest = valid.loc[valid["time_seconds"].idxmin()]
    fastest_seconds = fastest["time_seconds"]
    median_seconds = valid["time_seconds"].median()

    return (
        fastest["time_text"],
        f"{fastest['time_text']} ({fastest_seconds:.2f} s)",
        fastest["vehicle"],
        f"{format_lap_time(median_seconds)} ({median_seconds:.2f} s)"
    )


# Overall figures (used for the map hover text)
fastest_time, _, fastest_vehicle, _ = summary_stats(data)


def stat_card(title, value, accent):
    return html.Div(
        [
            html.H4(
                title,
                style={
                    "margin": "0 0 6px 0",
                    "color": "#5B6A70",
                    "fontWeight": "600"
                }
            ),
            html.H2(
                str(value),
                style={"margin": "0", "color": ASPHALT}
            )
        ],
        style={
            "backgroundColor": "white",
            "padding": "15px 18px",
            "borderRadius": "6px",
            "borderLeft": f"6px solid {accent}",
            "boxShadow": "0 2px 5px rgba(31,42,48,0.12)"
        }
    )


def stat_cards(df):
    """The four summary cards for the given (filtered) records."""
    _, fastest_display, fastest_vehicle, median_time = summary_stats(df)

    return [
        stat_card("Total Records", f"{len(df):,}", STAT_ACCENTS[0]),
        stat_card("Fastest Lap", fastest_display, STAT_ACCENTS[1]),
        stat_card("Fastest Vehicle", fastest_vehicle, STAT_ACCENTS[2]),
        stat_card("Median Lap Time", median_time, STAT_ACCENTS[3])
    ]


def panel(title, accent, children, margin_top="25px"):
    """White panel with a coloured heading rule."""
    return html.Div(
        [
            html.H2(
                title,
                style={
                    "color": ASPHALT,
                    "margin": "0 0 10px 0",
                    "paddingLeft": "12px",
                    "borderLeft": f"5px solid {accent}"
                }
            ),
            *children
        ],
        style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "6px",
            "boxShadow": "0 2px 5px rgba(31,42,48,0.12)",
            "marginTop": margin_top
        }
    )


def chart_panel(title, figure, accent, margin_top="25px", graph_id=None):
    """Panel containing a single chart."""
    return panel(
        title, accent, [dcc.Graph(id=graph_id, figure=figure)], margin_top
    )


# --------------------------------------------------
# TOP N LAP-TIME BAR CHART
# --------------------------------------------------

def nice_ticks(max_value):
    """Tick positions 0, step, 2 x step ... with about six ticks up to max_value."""
    for step in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20):
        if max_value / step <= 6:
            break
    return [round(i * step, 2) for i in range(int(max_value // step) + 1)]


def build_top_n_chart(df, n=20):
    plot_df = (
        df.nsmallest(n, "time_seconds")
        .sort_values("time_seconds")
        .reset_index(drop=True)
    )

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No laps match the filters.",
            showarrow=False
        )
        fig.update_layout(template="plotly_white", height=750)
        return fig

    # Seconds behind the fastest lap in view,
    # e.g. 404.85 - 390.70 = 14.15 seconds behind
    plot_df["gap_seconds"] = (
        plot_df["time_seconds"] - plot_df["time_seconds"].min()
    ).round(2)

    # The rank keeps every label unique
    plot_df["label"] = (
        "#"
        + plot_df["rank"].astype("Int64").astype(str)
        + " "
        + plot_df["vehicle"].astype(str)
        + " ("
        + plot_df["driver"].astype(str)
        + ")"
    )

    plot_df["position"] = range(len(plot_df))

  
    max_gap = plot_df["gap_seconds"].max()
    offset = max(max_gap * 0.15, 0.3)

    hover = (
        "Lap Time (seconds)=%{customdata[0]:.2f}<br>"
        "Vehicle (Driver)=%{customdata[1]}<br>"
        "time_text=%{customdata[2]}<br>"
        "driver=%{customdata[3]}<br>"
        "rank=%{customdata[4]}<br>"
        "Seconds behind the fastest lap=%{customdata[5]:.2f}"
        "<extra></extra>"
    )
    customdata = plot_df[
        ["time_seconds", "label", "time_text", "driver", "rank", "gap_seconds"]
    ].values

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["gap_seconds"] + offset,
            base=-offset,
            y=plot_df["position"],
            orientation="h",
            marker={"color": plot_df["time_seconds"], "coloraxis": "coloraxis"},
            customdata=customdata,
            hovertemplate=hover,
            showlegend=False
        )
    )

    # A dot at the end of every bar, in the same colours, with the lap time
    fig.add_trace(
        go.Scatter(
            x=plot_df["gap_seconds"],
            y=plot_df["position"],
            mode="markers+text",
            text=[
                f"{t} (fastest)" if gap == 0 else t
                for t, gap in zip(plot_df["time_text"], plot_df["gap_seconds"])
            ],
            textposition="middle right",
            cliponaxis=False,
            marker={
                "size": 12,
                "color": plot_df["time_seconds"],
                "coloraxis": "coloraxis",
                "opacity": 1,
                "line": {"width": 1, "color": "white"}
            },
            customdata=customdata,
            hovertemplate=hover,
            showlegend=False
        )
    )

    x_max = max(max_gap * 1.18, 1) + 1.5
    fig.update_yaxes(
        title_text="Vehicle (Driver)",
        tickmode="array",
        tickvals=plot_df["position"],
        ticktext=plot_df["label"],
        autorange="reversed"
    )
    fig.update_xaxes(
        title_text="Seconds behind the fastest lap in view",
        range=[-offset * 1.05, x_max],
        tickmode="array",
        tickvals=nice_ticks(x_max),
        zeroline=False
    )
    fig.update_layout(
        title=f"Top {len(plot_df)} Fastest Nordschleife Lap Times",
        coloraxis={
            "colorscale": SPEED_SCALE,
            "colorbar": {"title": "Lap time (s)<br>lower = faster"}
        },
        template="plotly_white",
        height=750
    )

    return fig


# --------------------------------------------------
# POWER-TO-WEIGHT SCATTER PLOT
# --------------------------------------------------

def build_power_to_weight_scatter(df):
    plot_df = df.dropna(
        subset=["power_to_weight", "time_seconds"]
    ).copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid power-to-weight data available.",
            showarrow=False
        )
        fig.update_layout(template="plotly_white")
        return fig

    fig = px.scatter(
        plot_df,
        x="power_to_weight",
        y="time_seconds",
        color="time_seconds",
        color_continuous_scale=SPEED_SCALE,
        hover_name="vehicle",
        hover_data=["driver", "time_text"],
        title="Power-to-Weight Ratio vs Lap Time",
        labels={
            "power_to_weight": "Power-to-Weight Ratio (PS/kg)",
            "time_seconds": "Lap Time (seconds)"
        }
    )

    fig.update_traces(
        marker={
            "size": 10,
            "opacity": 0.85,
            "line": {"width": 1, "color": "white"}
        }
    )

    fig.update_layout(
        coloraxis_colorbar={
            "title": "Lap time (s)<br>lower = faster"
        },
        template="plotly_white"
    )
    return fig


# --------------------------------------------------
# SPIDER CHART: TOP FIVE VEHICLES
# --------------------------------------------------

def minmax_score(series):
    """Normalize values to a 0-100 scale."""
    low = series.min()
    high = series.max()

    if high == low:
        return pd.Series(100.0, index=series.index)

    return (series - low) / (high - low) * 100


def build_spider_chart(df, n=5):
    required = [
        "time_seconds", "power_ps",
        "weight_kg", "power_to_weight"
    ]

    plot_df = df.dropna(subset=required).copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No complete performance data available.",
            showarrow=False
        )
        fig.update_layout(template="plotly_white")
        return fig

    # Higher score always means better relative performance
    plot_df["lap_score"] = minmax_score(
        -plot_df["time_seconds"]
    )
    plot_df["power_score"] = minmax_score(
        plot_df["power_ps"]
    )
    plot_df["weight_score"] = minmax_score(
        -plot_df["weight_kg"]
    )
    plot_df["ratio_score"] = minmax_score(
        plot_df["power_to_weight"]
    )

    top = plot_df.nsmallest(n, "time_seconds")

    categories = [
        "Lap Time", "Power", "Weight", "Power-to-Weight"
    ]

    fig = go.Figure()

    for i, (_, row) in enumerate(top.iterrows()):
        scores = [
            row["lap_score"],
            row["power_score"],
            row["weight_score"],
            row["ratio_score"]
        ]

        label = str(row["vehicle"])

        if pd.notna(row["rank"]):
            label += f" (Rank {int(row['rank'])})"

        color = TRACE_COLORS[i % len(TRACE_COLORS)]

        fig.add_trace(
            go.Scatterpolar(
                r=scores + scores[:1],
                theta=categories + categories[:1],
                fill="toself",
                name=label,
                line={"color": color, "width": 2},
                fillcolor=color,
                opacity=0.35
            )
        )

    fig.update_layout(
        title=f"Car Performance Comparison: Top {n}",
        legend={"title": {"text": "Vehicle<br>(lower rank = faster)"}},
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": "#D5DBD9"
            },
            "angularaxis": {"gridcolor": "#D5DBD9"}
        },
        template="plotly_white"
    )

    return fig


# --------------------------------------------------
# INTERACTIVE MAP
# --------------------------------------------------

# Approximate location of the Nürburgring complex
RING_LAT = 50.3356
RING_LON = 6.9475

# Plotly 6 renamed Scattermapbox -> Scattermap
NEW_MAPS = hasattr(go, "Scattermap")
MapTrace = go.Scattermap if NEW_MAPS else go.Scattermapbox
MAP_KEY = "map" if NEW_MAPS else "mapbox"

MAP_STYLES = {
    "Street": "open-street-map",
    "Light": "carto-positron",
    "Dark": "carto-darkmatter",
}


def build_map(style="open-street-map"):
    fig = go.Figure(
        MapTrace(
            
            )
        )

    fig.update_layout(
        **{
            MAP_KEY: {
                "style": style,
                "center": {"lat": RING_LAT, "lon": RING_LON},
                "zoom": 14
            }
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=550,
        showlegend=False
    )

    return fig


# --------------------------------------------------
# FILTERS
# --------------------------------------------------

# Makes whose names are more than one word
MULTI_WORD_MAKES = [
    "Alfa Romeo", "Aston Martin", "Land Rover", "Rolls Royce",
    "De Tomaso", "Gardner Douglas", "Lynk & Co",
    "Mercedes AMG", "Mercedes Benz"
]


def get_manufacturer(vehicle):
    """Guess the manufacturer from the start of the vehicle name."""
    if pd.isna(vehicle):
        return None

    name = str(vehicle).strip()
    lowered = name.lower().replace("-", " ")

    for make in MULTI_WORD_MAKES:
        if lowered.startswith(make.lower()):
            # Treat Mercedes-AMG / Mercedes-Benz as one manufacturer
            return "Mercedes" if make.startswith("Mercedes") else make

    first = name.split()[0] if name else None

    if first and first.lower().startswith("mercedes"):
        return "Mercedes"

    return first


data["manufacturer"] = data["vehicle"].apply(get_manufacturer)


def option_list(column, manufacturers=None, show_counts=True):
    """Dropdown options for a column, optionally with record counts.

    Pass manufacturers to only list entries from those manufacturers.
    """
    df = data

    if manufacturers:
        df = df[df["manufacturer"].astype(str).isin(manufacturers)]

    counts = df[column].dropna().astype(str).value_counts()

    return [
        {
            "label": f"{name} ({count:,})" if show_counts else name,
            "value": name
        }
        for name, count in sorted(counts.items())
    ]


all_times = data["time_seconds"].dropna()

# Lap-time slider range in seconds, rounded out to the millisecond
# (shown to the user as M:SS.mmm)
if all_times.empty:
    TIME_MIN, TIME_MAX = 0.0, 1200.0
else:
    TIME_MIN = float(int(all_times.min() * 1000 // 1) / 1000)
    TIME_MAX = float(-(-(all_times.max() * 1000) // 1) / 1000)

TOP_N_OPTIONS = [10, 20, 30, 50]


def format_clock(seconds):
    """Format whole seconds as M:SS."""
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def time_readout(time_range):
    low, high = (round(value, 3) for value in time_range)
    return f"{format_lap_time(low)} \u2013 {format_lap_time(high)}"


def time_marks():
    """Slider marks at round M:SS values inside the range."""
    span = TIME_MAX - TIME_MIN
    step = 30 if span <= 240 else 60 if span <= 600 else 120

    first = int(-(-TIME_MIN // step))
    last = int(TIME_MAX // step)

    return {
        n * step: format_clock(n * step)
        for n in range(first, last + 1)
    }


def apply_filters(df, manufacturers, vehicles, drivers, time_range):
    """Filter by manufacturer, vehicle, driver and lap-time range.

    time_range is (low, high) in seconds.
    """
    out = df

    if manufacturers:
        out = out[out["manufacturer"].astype(str).isin(manufacturers)]

    if vehicles:
        out = out[out["vehicle"].astype(str).isin(vehicles)]

    if drivers:
        out = out[out["driver"].astype(str).isin(drivers)]

    if time_range:
        low, high = time_range

        if low > TIME_MIN or high < TIME_MAX:
            # Half-millisecond margin guards against float rounding
            out = out[
                out["time_seconds"].between(low - 0.0005, high + 0.0005)
            ]

    return out


def filter_label(text):
    return html.Label(
        text,
        style={
            "display": "block",
            "marginBottom": "4px",
            "fontWeight": "600",
            "color": "#5B6A70"
        }
    )


# --------------------------------------------------
# DASHBOARD LAYOUT
# --------------------------------------------------

app.layout = html.Div(
    [
        # Header banner in German flag colours: black, red and gold
        html.Div(
            [
                html.H1(
                    "Nürburgring Nordschleife Insights",
                    style={
                        "textAlign": "center",
                        "color": "white",
                        "margin": "0",
                        "padding": "28px 0",
                        "backgroundColor": "#000000"
                    }
                ),
                # Three clearly visible horizontal stripes of the German flag
                html.Div(
                    style={
                        "height": "36px",
                        "background": (
                            "linear-gradient(to bottom, "
                            "#000000 0%, #000000 33.33%, "
                            "#DD0000 33.33%, #DD0000 66.66%, "
                            "#FFCC00 66.66%, #FFCC00 100%)"
                        )
                    }
                )
            ],
            style={
                "backgroundColor": "#000000",
                "borderRadius": "6px",
                "overflow": "hidden",
                "marginBottom": "25px"
            }
        ),

        panel(
            "Filters",
            ASPHALT,
            [
                html.Div(
                    [
                        html.Div([
                            filter_label("Manufacturer"),
                            dcc.Dropdown(
                                id="filter-manufacturer",
                                options=option_list(
                                    "manufacturer", show_counts=False
                                ),
                                multi=True,
                                placeholder="All manufacturers"
                            )
                        ]),
                        html.Div([
                            filter_label("Vehicle"),
                            dcc.Dropdown(
                                id="filter-vehicle",
                                options=option_list(
                                    "vehicle", show_counts=False
                                ),
                                value=[],
                                multi=True,
                                placeholder="All vehicles"
                            )
                        ]),
                        html.Div([
                            filter_label("Driver"),
                            dcc.Dropdown(
                                id="filter-driver",
                                options=option_list(
                                    "driver", show_counts=False
                                ),
                                multi=True,
                                placeholder="All drivers"
                            )
                        ]),
                        html.Div([
                            filter_label("Leaderboard size"),
                            dcc.Dropdown(
                                id="filter-top-n",
                                options=[
                                    {"label": f"Top {n}", "value": n}
                                    for n in TOP_N_OPTIONS
                                ],
                                value=20,
                                clearable=False
                            )
                        ])
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "2fr 3fr 2fr 1fr",
                        "gap": "15px"
                    }
                ),
                html.Div(
                    [
                        filter_label("Lap time (M:SS.mmm)"),
                        html.Div(
                            time_readout([TIME_MIN, TIME_MAX]),
                            id="time-readout",
                            style={
                                "marginBottom": "6px",
                                "fontWeight": "600",
                                "color": ASPHALT
                            }
                        ),
                        dcc.RangeSlider(
                            id="filter-time",
                            min=TIME_MIN,
                            max=TIME_MAX,
                            step=0.001,
                            value=[TIME_MIN, TIME_MAX],
                            marks=time_marks()
                        )
                    ],
                    style={"marginTop": "15px"}
                )
            ],
            margin_top="0"
        ),

        html.Div(
            stat_cards(data),
            id="stats",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                "gap": "15px",
                "marginTop": "25px"
            }
        ),

        chart_panel(
            "Lap Time Leaderboard",
            build_top_n_chart(data, 20),
            FOREST,
            graph_id="top-chart"
        ),
        html.Div(
            [
                chart_panel(
                    "Power-to-Weight Analysis",
                    build_power_to_weight_scatter(data),
                    AMBER,
                    margin_top="0",
                    graph_id="scatter-chart"
                ),
                chart_panel(
                    "Car Performance Comparison",
                    build_spider_chart(data, 5),
                    KERB_RED,
                    margin_top="0",
                    graph_id="spider-chart"
                )
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                "gap": "25px",
                "marginTop": "25px"
            }
        ),

        panel(
            "Track Map",
            STEEL,
            [
                dcc.Dropdown(
                    id="map-style",
                    options=[
                        {"label": name, "value": value}
                        for name, value in MAP_STYLES.items()
                    ],
                    value="open-street-map",
                    clearable=False,
                    style={"width": "220px", "marginBottom": "12px"}
                ),
                dcc.Graph(
                    id="track-map",
                    figure=build_map(),
                    config={"scrollZoom": True}
                )
            ]
        )
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "25px",
        "backgroundColor": TARMAC,
        "minHeight": "100vh"
    }
)


@app.callback(
    Output("stats", "children"),
    Output("top-chart", "figure"),
    Output("scatter-chart", "figure"),
    Output("spider-chart", "figure"),
    Input("filter-manufacturer", "value"),
    Input("filter-vehicle", "value"),
    Input("filter-driver", "value"),
    Input("filter-top-n", "value"),
    Input("filter-time", "value")
)
def update_dashboard(manufacturers, vehicles, drivers, top_n, time_range):
    chart_df = apply_filters(
        data, manufacturers, vehicles, drivers, time_range
    )

    return (
        stat_cards(chart_df),
        build_top_n_chart(chart_df, top_n or 20),
        build_power_to_weight_scatter(chart_df),
        build_spider_chart(chart_df, 5)
    )


@app.callback(
    Output("time-readout", "children"),
    Input("filter-time", "value")
)
def update_time_readout(time_range):
    return time_readout(time_range)


@app.callback(
    Output("filter-vehicle", "options"),
    Output("filter-vehicle", "value"),
    Input("filter-manufacturer", "value"),
    State("filter-vehicle", "value")
)
def update_vehicle_options(manufacturers, selected):
    """Only offer vehicles from the chosen manufacturers."""
    options = option_list("vehicle", manufacturers, show_counts=False)
    valid = {option["value"] for option in options}
    kept = [v for v in (selected or []) if v in valid]

    return options, kept


@app.callback(
    Output("track-map", "figure"),
    Input("map-style", "value")
)
def update_map_style(style):
    return build_map(style)


if __name__ == "__main__":
    app.run(debug=False)

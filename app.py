import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] h2 {
        color: #f5f5f5;
        font-size: 1.45rem;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .sidebar-section-title {
        color: rgba(245, 245, 245, 0.78);
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 1.4rem;
        margin-bottom: 0.45rem;
    }

    [data-testid="stSidebar"] .sidebar-divider {
        border-bottom: 1px solid rgba(245, 245, 245, 0.16);
        margin: 1rem 0 0.2rem;
    }

    .start-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.14), rgba(34, 197, 94, 0.10));
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0 1.25rem;
        text-align: center;
    }

    .start-card h2 {
        margin: 0 0 0.45rem;
        font-size: 1.45rem;
    }

    .start-card p {
        margin: 0;
        color: rgba(229, 231, 235, 0.86);
    }

    .hyperparameter-card {
        background: rgba(15, 23, 42, 0.48);
        border: 1px solid rgba(14, 165, 233, 0.28);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0 1rem;
    }

    .hyperparameter-card h3 {
        margin: 0 0 0.4rem;
    }

    .hyperparameter-card p {
        margin: 0;
        color: rgba(229, 231, 235, 0.86);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def sidebar_section(title):
    st.sidebar.markdown(
        f'<div class="sidebar-section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def sidebar_divider():
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


def load_selected_dataset(dataset_name):
    loaders = {
        "Iris": load_iris,
        "Breast Cancer Wisconsin": load_breast_cancer,
        "Wine": load_wine,
    }
    data = loaders[dataset_name](as_frame=True)
    df = data.frame.copy()
    target_name = data.target.name or "target"
    target_labels = dict(enumerate(data.target_names))
    df[target_name] = df[target_name].map(target_labels)
    return df, f"Built-in dataset: {dataset_name}"


def load_uploaded_dataset(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df, "Uploaded CSV dataset"
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_table(uploaded_file)
            return df, "Uploaded table dataset"
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}") from e


def resolve_target_name(df, target_name=None):
    if df.shape[1] < 2:
        raise ValueError("The dataset must include at least one feature column and one target column.")

    if target_name is None:
        target_name = df.columns[-1]

    if target_name not in df.columns:
        raise ValueError("Please select a valid target variable.")

    return target_name


def preprocess_data(df, target_name=None, high_cardinality_threshold=15):
    target_name = resolve_target_name(df, target_name)
    X = df.drop(columns=[target_name])
    y = df[target_name]
    original_feature_count = X.shape[1]

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_cardinality_cols = [
        column
        for column in categorical_cols
        if X[column].nunique(dropna=True) > high_cardinality_threshold
    ]
    X = X.drop(columns=high_cardinality_cols)

    model_data = pd.concat([X, y.rename(target_name)], axis=1).dropna()
    rows_dropped_missing = df.shape[0] - model_data.shape[0]
    X = model_data.drop(columns=[target_name]).copy()
    y = model_data[target_name]
    binary_feature_cols = [
        column
        for column in X.columns
        if X[column].dropna().isin([0, 1, True, False]).all()
    ]
    binary_feature_ratio = len(binary_feature_cols) / max(X.shape[1], 1)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    label_encoders = {}
    for column in categorical_cols:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].astype(str))
        label_encoders[column] = encoder

    scaler = None
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X = X.apply(pd.to_numeric, errors="coerce")
    model_data = pd.concat([X, y.rename(target_name)], axis=1).dropna()
    X = model_data.drop(columns=[target_name])
    y = model_data[target_name]

    if X.empty or y.empty:
        raise ValueError("No usable rows remain after preprocessing.")

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    preprocessing_info = {
        "original_feature_count": original_feature_count,
        "final_feature_count": X.shape[1],
        "high_cardinality_threshold": high_cardinality_threshold,
        "dropped_high_cardinality_cols": high_cardinality_cols,
        "retained_feature_columns": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "binary_feature_cols": binary_feature_cols,
        "binary_feature_ratio": binary_feature_ratio,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "rows_dropped_missing": rows_dropped_missing,
        "rows_used": len(X),
    }

    return X, y, target_name, preprocessing_info


def prepare_features(df, target_name=None):
    X, y, resolved_target_name, _ = preprocess_data(df, target_name)
    return X, y, resolved_target_name


MODEL_CLASSES = {
    "BernoulliNB": BernoulliNB,
    "GaussianNB": GaussianNB,
    "MultinomialNB": MultinomialNB,
}


def recommend_nb_classifier(X, preprocessing_info=None):
    if preprocessing_info and preprocessing_info["binary_feature_ratio"] >= 0.95:
        return (
            "BernoulliNB",
            "Reasoning: The retained raw features are almost entirely binary, which fits BernoulliNB well.",
        )

    values = X.to_numpy(dtype=float)
    binary_proportion = np.isin(values, [0, 1]).mean()
    has_negative_values = bool((values < 0).any())
    integer_like = bool(np.allclose(values, np.round(values)))
    non_negative = bool((values >= 0).all())

    if binary_proportion >= 0.95:
        return (
            "BernoulliNB",
            "Reasoning: Most model features are binary indicators, which fits BernoulliNB well.",
        )
    if has_negative_values:
        return (
            "GaussianNB",
            "Reasoning: Negative values were detected, and GaussianNB can handle continuous signed features.",
        )
    if non_negative and integer_like:
        return (
            "MultinomialNB",
            "Reasoning: Features are discrete and non-negative, which is suitable for MultinomialNB.",
        )
    return (
        "GaussianNB",
        "Reasoning: Continuous numeric features were detected, making GaussianNB the safest default.",
    )


def build_nb_model(model_name, model_params):
    if model_name == "GaussianNB":
        return GaussianNB(var_smoothing=model_params["var_smoothing"])
    if model_name == "MultinomialNB":
        return MultinomialNB(
            alpha=model_params["alpha"],
            fit_prior=model_params["fit_prior"],
        )
    if model_name == "BernoulliNB":
        return BernoulliNB(
            alpha=model_params["alpha"],
            fit_prior=model_params["fit_prior"],
            binarize=model_params["binarize"],
        )
    raise ValueError(f"Unsupported model type: {model_name}")


def validate_model_choice(X, model_name):
    if model_name == "MultinomialNB" and (X < 0).any(axis=None):
        raise ValueError("MultinomialNB requires non-negative feature values.")


def build_model_parameter_controls(model_name, container=st.sidebar, show_sidebar_divider=True):
    model_params = {}

    container.markdown(
        """
        <div class="hyperparameter-card">
            <h3>Hyperparameter Tuning</h3>
            <p>
                Hyperparameters are the steering wheel of your model. Tweak these
                values to change how the algorithm learns; the metrics and charts
                update on the next Streamlit rerun.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_name == "GaussianNB":
        container.write("### Variance Smoothing")
        container.caption("Widens the Gaussian curve slightly so unseen or rare values are less brittle.")
        smoothing_power = container.slider(
            "Variance Smoothing (log10)",
            min_value=-11,
            max_value=-1,
            value=-9,
            step=1,
            help="GaussianNB var_smoothing value is 10 raised to this exponent.",
        )
        model_params["var_smoothing"] = 10.0 ** smoothing_power
        container.caption(f"Current var_smoothing = {model_params['var_smoothing']:.0e}")
    elif model_name == "MultinomialNB":
        container.write("### Alpha")
        container.caption("Prevents zero-probability errors when a feature/class combination is rare.")
        model_params["alpha"] = container.slider(
            "Alpha (Additive Smoothing)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )
        container.write("### Fit Prior")
        container.caption("Learns how common each class is before looking at the feature values.")
        model_params["fit_prior"] = container.checkbox(
            "Fit Prior",
            value=True,
            help="Whether to learn class prior probabilities.",
        )
    elif model_name == "BernoulliNB":
        container.write("### Alpha")
        container.caption("Prevents zero-probability errors when a binary feature/class combination is rare.")
        model_params["alpha"] = container.slider(
            "Alpha (Additive Smoothing)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )
        container.write("### Fit Prior")
        container.caption("Learns how common each class is before looking at the feature values.")
        model_params["fit_prior"] = container.checkbox(
            "Fit Prior",
            value=True,
            help="Whether to learn class prior probabilities.",
        )
        container.write("### Binarize")
        container.caption("Turns values above this threshold into 1 and values at/below it into 0.")
        model_params["binarize"] = container.number_input(
            "Binarize",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Threshold for binarizing sample features.",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    if show_sidebar_divider:
        sidebar_divider()
    return model_params


def train_and_evaluate(X, y, model_name, model_params):
    validate_model_choice(X, model_name)
    model = build_nb_model(model_name, model_params)
    class_count = y.nunique()
    test_count = int(np.ceil(len(y) * 0.25))
    can_split = (
        len(y) >= 4
        and y.value_counts().min() >= 2
        and test_count >= class_count
        and len(y) - test_count >= class_count
    )

    if can_split:
        X_train, X_eval, y_train, y_eval = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        accuracy_label = "Test Accuracy"
    else:
        X_train, X_eval, y_train, y_eval = X, X, y, y
        accuracy_label = "Training Accuracy"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)

    return model, X_eval, y_eval, y_pred, accuracy, accuracy_label


def build_gaussian_plot(X, y, target_name, var_smoothing):
    feature_name = X.columns[0]
    feature_values = X[feature_name]
    feature_min = feature_values.min()
    feature_max = feature_values.max()
    value_range = feature_max - feature_min
    if value_range == 0:
        value_range = 1.0

    x_axis = np.linspace(
        feature_min - value_range * 0.1,
        feature_max + value_range * 0.1,
        200,
    )
    smoothing_variance = var_smoothing * max(feature_values.var(ddof=0), 1e-9)
    plot_rows = []

    for class_name in y.unique():
        class_values = X.loc[y == class_name, feature_name]
        mean = class_values.mean()
        variance = max(class_values.var(ddof=0) + smoothing_variance, 1e-9)
        density = np.exp(-0.5 * ((x_axis - mean) ** 2) / variance)
        density = density / np.sqrt(2 * np.pi * variance)

        for feature_value, probability_density in zip(x_axis, density):
            plot_rows.append(
                {
                    feature_name: feature_value,
                    "Probability Density": probability_density,
                    target_name: class_name,
                }
            )

    return px.line(
        pd.DataFrame(plot_rows),
        x=feature_name,
        y="Probability Density",
        color=target_name,
        title=f"Gaussian Distribution Plot for {feature_name}",
    )


def build_confusion_matrix_plot(y_test, y_pred):
    labels = sorted(pd.Series(y_test).astype(str).unique())
    matrix = confusion_matrix(
        pd.Series(y_test).astype(str),
        pd.Series(y_pred).astype(str),
        labels=labels,
    )
    return px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix",
    )


def create_preprocessing_diagram(df, target_name, preprocessing_info):
    feature_df = df.drop(columns=[target_name])
    missing_values = int(df.isna().sum().sum())
    categorical_columns = preprocessing_info["categorical_cols"]
    numeric_columns = preprocessing_info["numeric_cols"]
    dropped_rows = preprocessing_info["rows_dropped_missing"]
    high_cardinality_cols = preprocessing_info["dropped_high_cardinality_cols"]

    steps = [
        {
            "label": "Input Data Loading",
            "status": "active",
            "detail": f"{df.shape[0]:,} rows and {df.shape[1]:,} columns loaded.",
        },
        {
            "label": "Target Column Separation",
            "status": "active",
            "detail": f"Target column: {target_name}. Feature columns: {feature_df.shape[1]:,}.",
        },
        {
            "label": "Missing Value Handling",
            "status": "active" if missing_values else "skip",
            "detail": (
                f"{missing_values:,} missing values found. {dropped_rows:,} rows removed after preprocessing."
                if missing_values
                else "No missing values detected."
            ),
        },
        {
            "label": "High-Cardinality Filter",
            "status": "active" if high_cardinality_cols else "skip",
            "detail": (
                "Dropped: " + ", ".join(high_cardinality_cols)
                if high_cardinality_cols
                else "No high-cardinality categorical columns exceeded the threshold."
            ),
        },
        {
            "label": "Categorical Encoding",
            "status": "active" if categorical_columns else "skip",
            "detail": (
                f"Label encoded: {', '.join(categorical_columns)}."
                if categorical_columns
                else "No categorical feature columns detected."
            ),
        },
        {
            "label": "Numerical Scaling",
            "status": "active" if numeric_columns else "skip",
            "detail": (
                f"Scaled numeric columns: {', '.join(numeric_columns[:6])}"
                + ("..." if len(numeric_columns) > 6 else ".")
                if numeric_columns
                else "No numeric feature columns detected before encoding."
            ),
        },
        {
            "label": "Final Model Matrix",
            "status": "active",
            "detail": (
                f"Original features: {preprocessing_info['original_feature_count']:,}. "
                f"Final encoded features: {preprocessing_info['final_feature_count']:,}."
            ),
        },
    ]

    colors = {
        "active": "#6ee7b7",
        "skip": "#94a3b8",
    }
    x_values = list(range(len(steps)))
    y_values = [0] * len(steps)

    def wrap_step_label(label, max_chars=16):
        words = label.split()
        lines = []
        current_line = []

        for word in words:
            candidate = " ".join(current_line + [word])
            if len(candidate) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return "<br>".join(lines)

    fig = go.Figure()
    for index in range(len(steps) - 1):
        fig.add_trace(
            go.Scatter(
                x=[x_values[index], x_values[index + 1]],
                y=[0, 0],
                mode="lines",
                line=dict(width=2, color="rgba(148, 163, 184, 0.55)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            marker=dict(
                size=34,
                color=[colors[step["status"]] for step in steps],
                line=dict(width=2, color="rgba(15, 23, 42, 0.95)"),
            ),
            text=[f"{index + 1}" for index in range(len(steps))],
            textfont=dict(size=13, color="#0f172a"),
            customdata=[[step["label"], step["detail"], step["status"].title()] for step in steps],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Status: %{customdata[2]}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    for index, step in enumerate(steps):
        fig.add_annotation(
            x=x_values[index],
            y=-0.32,
            text=wrap_step_label(step["label"]),
            showarrow=False,
            font=dict(size=12, color="#e5e7eb"),
            align="center",
            width=125,
        )

    fig.update_layout(
        title=dict(text="Preprocessing Pipeline", x=0.5),
        height=360,
        margin=dict(l=30, r=30, t=70, b=120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        xaxis=dict(visible=False, range=[-0.45, len(steps) - 0.55]),
        yaxis=dict(visible=False, range=[-0.7, 0.45]),
    )
    return fig


def build_prediction_input(df, target_name, preprocessing_info, container=st.sidebar):
    feature_columns = [
        column
        for column in df.columns
        if column != target_name
        and column not in preprocessing_info["dropped_high_cardinality_cols"]
    ]
    container.header("Prediction Input")
    input_data = {}

    if preprocessing_info["dropped_high_cardinality_cols"]:
        container.caption(
            "Skipped high-cardinality fields: "
            + ", ".join(preprocessing_info["dropped_high_cardinality_cols"])
        )

    for column in feature_columns:
        series = df[column].dropna()
        if series.empty:
            input_data[column] = container.text_input(
                f"Enter {column}:",
                value="",
                key=f"predict_{column}",
            )
        elif pd.api.types.is_numeric_dtype(series):
            min_value = float(series.min())
            max_value = float(series.max())
            mean_value = float(series.mean())
            input_data[column] = container.number_input(
                f"Enter {column}:",
                min_value=min_value,
                max_value=max_value,
                value=mean_value,
                step=0.1,
                key=f"predict_{column}",
            )
        elif pd.api.types.is_bool_dtype(series):
            input_data[column] = container.selectbox(
                f"Enter {column}:",
                options=sorted(series.unique().tolist()),
                key=f"predict_{column}",
            )
        else:
            input_data[column] = container.selectbox(
                f"Enter {column}:",
                options=sorted(series.astype(str).unique().tolist()),
                key=f"predict_{column}",
            )

    return input_data


def preprocess_prediction_input(input_data, preprocessing_info):
    new_df = pd.DataFrame([input_data])

    for column in preprocessing_info["categorical_cols"]:
        if column not in new_df.columns:
            continue

        encoder = preprocessing_info["label_encoders"][column]
        value = str(new_df.at[0, column])
        if value not in encoder.classes_:
            raise ValueError(f"Unknown category for {column}: {value}")
        new_df[column] = encoder.transform([value])

    for column in preprocessing_info["numeric_cols"]:
        if column not in new_df.columns:
            continue
        new_df[column] = pd.to_numeric(new_df[column], errors="coerce")

    if preprocessing_info["numeric_cols"] and preprocessing_info["scaler"] is not None:
        new_df[preprocessing_info["numeric_cols"]] = preprocessing_info["scaler"].transform(
            new_df[preprocessing_info["numeric_cols"]]
        )

    new_df = new_df.reindex(columns=preprocessing_info["retained_feature_columns"])

    if new_df.isna().any(axis=None):
        raise ValueError("Prediction input contains values that could not be converted.")

    return new_df


STEP_DESCRIPTIONS = {
    "Input Data Loading": "Loads the raw dataset from either the built-in dataset picker or the uploaded file.",
    "Target Column Separation": "Splits the selected target variable away from the feature columns used for training.",
    "Missing Value Handling": "Removes rows that contain missing or non-convertible values after preprocessing.",
    "High-Cardinality Filter": "Drops categorical fields with too many unique values before encoding.",
    "Categorical Encoding": "Converts remaining text or categorical feature values into numeric label codes.",
    "Numerical Scaling": "Standardizes numeric features so they have comparable scale.",
    "Final Model Matrix": "Builds the final numeric feature matrix that Gaussian Naive Bayes receives.",
}

STEP_JUSTIFICATIONS = {
    "Input Data Loading": "The model needs a structured table before any learning or evaluation can happen.",
    "Target Column Separation": "Features describe each row, while the target is the class the model learns to predict.",
    "Missing Value Handling": "Gaussian Naive Bayes cannot train on NaN values, so unusable rows must be removed.",
    "High-Cardinality Filter": "Dropping identifier-like categorical fields prevents inflated feature counts and noisy encodings.",
    "Categorical Encoding": "Scikit-learn estimators require numeric inputs, so text categories need numeric representation.",
    "Numerical Scaling": "Scaling prevents large numeric ranges from dominating the model diagnostics and comparisons.",
    "Final Model Matrix": "This confirms the exact shape and columns that are passed into training and prediction.",
}


def get_preprocessing_steps():
    return list(STEP_DESCRIPTIONS.keys())


def build_raw_feature_frame(df, target_name):
    return df.drop(columns=[target_name])


def build_filtered_feature_frame(df, target_name, preprocessing_info):
    raw_features = build_raw_feature_frame(df, target_name)
    return raw_features.drop(columns=preprocessing_info["dropped_high_cardinality_cols"])


def build_encoded_feature_frame(df, target_name, preprocessing_info):
    encoded_features = build_filtered_feature_frame(df, target_name, preprocessing_info).dropna().copy()

    for column in preprocessing_info["categorical_cols"]:
        if column in encoded_features.columns:
            encoder = preprocessing_info["label_encoders"][column]
            encoded_features[column] = encoder.transform(encoded_features[column].astype(str))

    return encoded_features


def build_scaled_feature_frame(df, target_name, preprocessing_info):
    scaled_features = build_encoded_feature_frame(df, target_name, preprocessing_info)

    if preprocessing_info["numeric_cols"] and preprocessing_info["scaler"] is not None:
        scaled_features[preprocessing_info["numeric_cols"]] = preprocessing_info["scaler"].transform(
            scaled_features[preprocessing_info["numeric_cols"]]
        )

    return scaled_features.apply(pd.to_numeric, errors="coerce")


def get_step_frames(step, df, target_name, X, y, preprocessing_info):
    raw_features = build_raw_feature_frame(df, target_name)
    filtered_features = build_filtered_feature_frame(df, target_name, preprocessing_info)
    encoded_features = build_encoded_feature_frame(df, target_name, preprocessing_info)
    scaled_features = build_scaled_feature_frame(df, target_name, preprocessing_info)
    model_ready = pd.concat([X, y.rename(target_name)], axis=1)

    if step == "Input Data Loading":
        return df, df
    if step == "Target Column Separation":
        return df, raw_features
    if step == "Missing Value Handling":
        return df, df.dropna()
    if step == "High-Cardinality Filter":
        return raw_features, filtered_features
    if step == "Categorical Encoding":
        return filtered_features, encoded_features
    if step == "Numerical Scaling":
        return encoded_features, scaled_features
    if step == "Final Model Matrix":
        return df, model_ready
    return df, df


def create_step_impact_chart(step, df, target_name, before_df, after_df, preprocessing_info):
    feature_df = build_raw_feature_frame(df, target_name)

    if step == "Input Data Loading":
        summary = pd.DataFrame(
            {
                "Measure": ["Rows", "Columns"],
                "Count": [df.shape[0], df.shape[1]],
            }
        )
        return px.bar(summary, x="Measure", y="Count", title="Loaded Dataset Size")

    if step == "Target Column Separation":
        counts = pd.Series(
            {
                "Feature Columns": feature_df.shape[1],
                "Target Columns": 1,
            }
        ).reset_index()
        counts.columns = ["Column Type", "Count"]
        return px.bar(counts, x="Column Type", y="Count", title="Feature vs Target Split")

    if step == "Missing Value Handling":
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(10)
        if missing.empty:
            missing = pd.Series({"No Missing Values": 0})
        missing_df = missing.reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        return px.bar(missing_df, x="Column", y="Missing Values", title="Missing Values by Column")

    if step == "High-Cardinality Filter":
        dropped_columns = preprocessing_info["dropped_high_cardinality_cols"]
        if not dropped_columns:
            cardinality_df = pd.DataFrame({"Column": ["No columns dropped"], "Unique Values": [0]})
        else:
            cardinality_df = pd.DataFrame(
                {
                    "Column": dropped_columns,
                    "Unique Values": [feature_df[column].nunique(dropna=True) for column in dropped_columns],
                }
            )
        return px.bar(
            cardinality_df,
            x="Column",
            y="Unique Values",
            title="Dropped High-Cardinality Columns",
        )

    if step == "Categorical Encoding":
        categorical_columns = preprocessing_info["categorical_cols"]
        if not categorical_columns:
            encoded_counts = pd.DataFrame({"Column": ["No categorical columns"], "Encoded Columns": [0]})
        else:
            encoded_counts = pd.DataFrame(
                {
                    "Column": categorical_columns,
                    "Label Count": [
                        feature_df[column].dropna().astype(str).nunique()
                        for column in categorical_columns
                    ],
                }
            )
        return px.bar(encoded_counts, x="Column", y=encoded_counts.columns[-1], title="Label Encoded Columns")

    if step == "Numerical Scaling":
        numeric_columns = preprocessing_info["numeric_cols"]
        if numeric_columns:
            melted = after_df[numeric_columns[:6]].melt(var_name="Feature", value_name="Scaled Value")
            return px.box(melted, x="Feature", y="Scaled Value", title="Scaled Numeric Feature Distributions")
        return px.bar(
            pd.DataFrame({"Measure": ["Numeric Columns"], "Count": [0]}),
            x="Measure",
            y="Count",
            title="Numeric Columns",
        )

    if step == "Final Model Matrix":
        summary = pd.DataFrame(
            {
                "Measure": ["Rows", "Model Features", "Target Classes"],
                "Count": [len(X), len(X.columns), y.nunique()],
            }
        )
        return px.bar(summary, x="Measure", y="Count", title="Final Model Matrix Summary")

    return None


def show_step_details(step, df, target_name, X, y, preprocessing_info):
    before_df, after_df = get_step_frames(step, df, target_name, X, y, preprocessing_info)

    st.subheader(f"Step Details: {step}")
    st.write(f"**What this does:** {STEP_DESCRIPTIONS[step]}")
    st.write(f"**Why the model needs this:** {STEP_JUSTIFICATIONS[step]}")

    before_col, after_col = st.columns(2)
    with before_col:
        st.caption("Before")
        st.dataframe(before_df.head(5), width="stretch")
    with after_col:
        st.caption("After")
        st.dataframe(after_df.head(5), width="stretch")

    fig = create_step_impact_chart(step, df, target_name, before_df, after_df, preprocessing_info)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")


st.title("Naive Bayes Visualization App")
st.write("Explore the performance of Naive Bayes with interactive visualizations.")
st.markdown(
    """
    <div class="start-card">
        <h2>Welcome!</h2>
        <p>
            Follow the tabs below to see how your data moves from raw rows,
            through preprocessing, into a Naive Bayes model, and finally into
            an interactive prediction playground.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Settings")

sidebar_section("Data Source")
data_source_choice = st.sidebar.radio(
    "Data Source",
    ("Use Built-in Dataset", "Upload Custom Dataset"),
    label_visibility="collapsed",
)
sidebar_divider()

dataset = None
uploaded_file = None
if data_source_choice == "Use Built-in Dataset":
    sidebar_section("Select Dataset")
    dataset = st.sidebar.selectbox(
        "Select Dataset",
        ["Iris", "Breast Cancer Wisconsin", "Wine"],
        label_visibility="collapsed",
    )
    sidebar_divider()
else:
    sidebar_section("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv", "txt"],
        label_visibility="collapsed",
    )
    sidebar_divider()

try:
    selected_target_col = None
    if data_source_choice == "Upload Custom Dataset":
        if uploaded_file is None:
            st.info("Upload a CSV or TXT dataset to begin.")
            st.stop()

        df, data_source = load_uploaded_dataset(uploaded_file)
        st.success("Dataset loaded successfully!")
        column_names = df.columns.tolist()
        st.success(f"Found {len(column_names)} columns.")

        sidebar_section("Select Target Variable")
        selected_target_col = st.sidebar.selectbox(
            "Select Target Variable",
            options=[""] + column_names,
            index=0,
            label_visibility="collapsed",
        )
        sidebar_divider()

        if selected_target_col:
            st.success(f"Target variable set to: {selected_target_col}")
            original_columns = df.columns.tolist()
            st.success(f"Original columns: {original_columns}")
        else:
            st.warning("Please select a target variable.")
            st.stop()
    else:
        df, data_source = load_selected_dataset(dataset)
        st.success(f"Using the {dataset} dataset.")

    X, y, target_name, preprocessing_info = preprocess_data(df, selected_target_col)
    recommended_model_name, model_justification = recommend_nb_classifier(X, preprocessing_info)

    sidebar_section("Model Recommendation")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Auto", "BernoulliNB", "MultinomialNB", "GaussianNB"],
        label_visibility="collapsed",
    )
    sidebar_divider()

    selected_model_name = recommended_model_name if model_type == "Auto" else model_type
    tab_data, tab_model, tab_predict = st.tabs(
        ["Data & Preprocessing", "Model Insights", "Playground / Predict"]
    )

    with tab_data:
        st.header("1. Choose Data Source")
        st.write(f"Active source: **{data_source_choice}**")
        st.write(f"Data source: **{data_source}**")
        st.write(f"Target column: **{target_name}**")

        st.header("Hyperparameter Tuning")
        model_params = build_model_parameter_controls(
            selected_model_name,
            container=st,
            show_sidebar_divider=False,
        )

    try:
        model, X_test, y_test, y_pred, accuracy, accuracy_label = train_and_evaluate(
            X,
            y,
            selected_model_name,
            model_params,
        )
    except ValueError as e:
        st.error(f"Selected model is incompatible with the current features: {e}")
        st.stop()

    st.success("Model trained successfully!")

    sidebar_section("Data Metrics")
    st.sidebar.write(f"**Original Features:** {preprocessing_info['original_feature_count']}")
    st.sidebar.write(f"**Final Encoded Features:** {preprocessing_info['final_feature_count']}")
    if preprocessing_info["dropped_high_cardinality_cols"]:
        st.sidebar.caption(
            "Dropped high-cardinality fields: "
            + ", ".join(preprocessing_info["dropped_high_cardinality_cols"])
        )
    sidebar_divider()

    sidebar_section("Pipeline Step Details")
    selected_step = st.sidebar.selectbox(
        "Pipeline Step Details",
        get_preprocessing_steps(),
        label_visibility="collapsed",
    )
    sidebar_divider()

    with tab_data:
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Original Features", f"{preprocessing_info['original_feature_count']:,}")
        metric_2.metric("Rows Used", f"{len(X):,}")
        metric_3.metric("Final Encoded Features", f"{len(X.columns):,}")

        st.header("2. Pipeline Steps")
        st.plotly_chart(
            create_preprocessing_diagram(df, target_name, preprocessing_info),
            width="stretch",
        )

        with st.expander("Detailed Step View", expanded=True):
            show_step_details(selected_step, df, target_name, X, y, preprocessing_info)

        with st.expander("How to read this tab"):
            st.write(
                "Start with the pipeline chart, then use the sidebar's Pipeline Step Details "
                "selector to inspect the before/after tables for each preprocessing step."
            )

        st.subheader("Data Preview")
        st.dataframe(df.head(10), width="stretch")

    with tab_model:
        st.header("3. Model Insights")

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric(accuracy_label, f"{accuracy:.2%}")
        metric_2.metric("Selected Model", selected_model_name)
        metric_3.metric("Target Classes", f"{y.nunique():,}")

        st.write(f"**Recommended Model:** {recommended_model_name}")
        st.write(model_justification)
        if model_type == "Auto":
            st.info(f"Using Auto selection: {selected_model_name}")
        else:
            st.success(f"Model override selected: {selected_model_name}")

        st.plotly_chart(
            build_gaussian_plot(X, y, target_name, model_params.get("var_smoothing", 1e-9)),
            width="stretch",
        )
        with st.expander("Explanation of Gaussian Distribution Plot"):
            st.write(
                "This chart shows how the first model-ready feature is distributed for each target class. "
                "More separated curves usually mean that feature is more useful for classification."
            )

        st.plotly_chart(build_confusion_matrix_plot(y_test, y_pred), width="stretch")
        with st.expander("Explanation of Confusion Matrix"):
            st.write(
                "A confusion matrix compares actual classes with predicted classes. "
                "Values on the diagonal are correct predictions; off-diagonal values are mistakes."
            )

    with tab_predict:
        st.header("4. Playground / Predict")
        st.write("Enter values for one new row and ask the trained model for a prediction.")

        input_data = build_prediction_input(df, target_name, preprocessing_info, container=st)

        if st.button("Predict"):
            try:
                new_data_df = preprocess_prediction_input(input_data, preprocessing_info)
                prediction = model.predict(new_data_df)[0]
                st.success(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        with st.expander("How to use the playground"):
            st.write(
                "The fields match the features retained after preprocessing. "
                "High-cardinality fields dropped by the pipeline are intentionally skipped."
            )
except Exception as e:
    st.warning(f"Error loading or processing the dataset: {e}")

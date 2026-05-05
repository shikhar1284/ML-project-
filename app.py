import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


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


def prepare_features(df, target_name=None):
    if df.shape[1] < 2:
        raise ValueError("The dataset must include at least one feature column and one target column.")

    if target_name is None:
        target_name = df.columns[-1]

    if target_name not in df.columns:
        raise ValueError("Please select a valid target variable.")

    X = df.drop(columns=[target_name])
    y = df[target_name]

    X = pd.get_dummies(X, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")

    model_data = pd.concat([X, y.rename(target_name)], axis=1).dropna()
    X = model_data.drop(columns=[target_name])
    y = model_data[target_name]

    if X.empty or y.empty:
        raise ValueError("No usable rows remain after preprocessing.")

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    return X, y, target_name


MODEL_CLASSES = {
    "BernoulliNB": BernoulliNB,
    "GaussianNB": GaussianNB,
    "MultinomialNB": MultinomialNB,
}


def recommend_nb_classifier(X):
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


def build_nb_model(model_name, var_smoothing):
    if model_name == "GaussianNB":
        return GaussianNB(var_smoothing=var_smoothing)
    if model_name == "MultinomialNB":
        return MultinomialNB()
    if model_name == "BernoulliNB":
        return BernoulliNB()
    raise ValueError(f"Unsupported model type: {model_name}")


def validate_model_choice(X, model_name):
    if model_name == "MultinomialNB" and (X < 0).any(axis=None):
        raise ValueError("MultinomialNB requires non-negative feature values.")


def train_and_evaluate(X, y, model_name, var_smoothing):
    validate_model_choice(X, model_name)
    model = build_nb_model(model_name, var_smoothing)
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


def create_preprocessing_diagram(df, target_name, processed_feature_count):
    feature_df = df.drop(columns=[target_name])
    missing_values = int(df.isna().sum().sum())
    categorical_columns = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    dropped_rows = int(df.shape[0] - df.dropna().shape[0])

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
            "label": "Categorical Encoding",
            "status": "active" if categorical_columns else "skip",
            "detail": (
                f"One-hot encoded: {', '.join(categorical_columns)}."
                if categorical_columns
                else "No categorical feature columns detected."
            ),
        },
        {
            "label": "Numerical Conversion",
            "status": "active" if numeric_columns else "skip",
            "detail": (
                f"Numeric columns retained/coerced: {', '.join(numeric_columns[:6])}"
                + ("..." if len(numeric_columns) > 6 else ".")
                if numeric_columns
                else "No numeric feature columns detected before encoding."
            ),
        },
        {
            "label": "Final Model Matrix",
            "status": "active",
            "detail": f"Final feature matrix contains {processed_feature_count:,} model-ready columns.",
        },
    ]

    colors = {
        "active": "#6ee7b7",
        "skip": "#94a3b8",
    }
    x_values = list(range(len(steps)))
    y_values = [0] * len(steps)

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
            y=-0.28,
            text=step["label"],
            showarrow=False,
            font=dict(size=12, color="#e5e7eb"),
            align="center",
            width=120,
        )

    fig.update_layout(
        title=dict(text="Preprocessing Pipeline", x=0.5),
        height=330,
        margin=dict(l=30, r=30, t=70, b=95),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        xaxis=dict(visible=False, range=[-0.45, len(steps) - 0.55]),
        yaxis=dict(visible=False, range=[-0.55, 0.45]),
    )
    return fig


def build_prediction_input(df, target_name):
    feature_columns = [column for column in df.columns if column != target_name]
    st.sidebar.header("Prediction Input")
    input_data = {}

    for column in feature_columns:
        series = df[column].dropna()
        if series.empty:
            input_data[column] = st.sidebar.text_input(
                f"Enter {column}:",
                value="",
                key=f"predict_{column}",
            )
        elif pd.api.types.is_numeric_dtype(series):
            min_value = float(series.min())
            max_value = float(series.max())
            mean_value = float(series.mean())
            input_data[column] = st.sidebar.number_input(
                f"Enter {column}:",
                min_value=min_value,
                max_value=max_value,
                value=mean_value,
                step=0.1,
                key=f"predict_{column}",
            )
        elif pd.api.types.is_bool_dtype(series):
            input_data[column] = st.sidebar.selectbox(
                f"Enter {column}:",
                options=sorted(series.unique().tolist()),
                key=f"predict_{column}",
            )
        else:
            input_data[column] = st.sidebar.selectbox(
                f"Enter {column}:",
                options=sorted(series.astype(str).unique().tolist()),
                key=f"predict_{column}",
            )

    return input_data


def preprocess_prediction_input(input_data, model_columns):
    new_df = pd.DataFrame([input_data])
    new_df = pd.get_dummies(new_df, drop_first=False)
    new_df = new_df.reindex(columns=model_columns, fill_value=0)
    new_df = new_df.apply(pd.to_numeric, errors="coerce")

    if new_df.isna().any(axis=None):
        raise ValueError("Prediction input contains values that could not be converted.")

    return new_df


STEP_DESCRIPTIONS = {
    "Input Data Loading": "Loads the raw dataset from either the built-in dataset picker or the uploaded file.",
    "Target Column Separation": "Splits the selected target variable away from the feature columns used for training.",
    "Missing Value Handling": "Removes rows that contain missing or non-convertible values after preprocessing.",
    "Categorical Encoding": "Converts text or categorical feature values into numeric one-hot indicator columns.",
    "Numerical Conversion": "Keeps numeric feature columns model-ready and coerces numeric-like values to numbers.",
    "Final Model Matrix": "Builds the final numeric feature matrix that Gaussian Naive Bayes receives.",
}

STEP_JUSTIFICATIONS = {
    "Input Data Loading": "The model needs a structured table before any learning or evaluation can happen.",
    "Target Column Separation": "Features describe each row, while the target is the class the model learns to predict.",
    "Missing Value Handling": "Gaussian Naive Bayes cannot train on NaN values, so unusable rows must be removed.",
    "Categorical Encoding": "Scikit-learn estimators require numeric inputs, so text categories need numeric representation.",
    "Numerical Conversion": "The model estimates numeric Gaussian distributions for each feature and class.",
    "Final Model Matrix": "This confirms the exact shape and columns that are passed into training and prediction.",
}


def get_preprocessing_steps():
    return list(STEP_DESCRIPTIONS.keys())


def build_raw_feature_frame(df, target_name):
    return df.drop(columns=[target_name])


def build_encoded_feature_frame(df, target_name):
    raw_features = build_raw_feature_frame(df, target_name)
    return pd.get_dummies(raw_features, drop_first=False)


def build_numeric_feature_frame(df, target_name):
    encoded_features = build_encoded_feature_frame(df, target_name)
    return encoded_features.apply(pd.to_numeric, errors="coerce")


def get_step_frames(step, df, target_name, X, y):
    raw_features = build_raw_feature_frame(df, target_name)
    encoded_features = build_encoded_feature_frame(df, target_name)
    numeric_features = build_numeric_feature_frame(df, target_name)
    model_ready = pd.concat([X, y.rename(target_name)], axis=1)

    if step == "Input Data Loading":
        return df, df
    if step == "Target Column Separation":
        return df, raw_features
    if step == "Missing Value Handling":
        return df, df.dropna()
    if step == "Categorical Encoding":
        return raw_features, encoded_features
    if step == "Numerical Conversion":
        return encoded_features, numeric_features
    if step == "Final Model Matrix":
        return df, model_ready
    return df, df


def create_step_impact_chart(step, df, target_name, before_df, after_df):
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

    if step == "Categorical Encoding":
        categorical_columns = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if not categorical_columns:
            encoded_counts = pd.DataFrame({"Column": ["No categorical columns"], "Encoded Columns": [0]})
        else:
            encoded_counts = pd.DataFrame(
                {
                    "Column": categorical_columns,
                    "Encoded Columns": [
                        feature_df[column].dropna().astype(str).nunique()
                        for column in categorical_columns
                    ],
                }
            )
        return px.bar(encoded_counts, x="Column", y="Encoded Columns", title="One-Hot Encoded Columns")

    if step == "Numerical Conversion":
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            melted = feature_df[numeric_columns[:6]].melt(var_name="Feature", value_name="Value")
            return px.box(melted, x="Feature", y="Value", title="Numeric Feature Distributions")
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


def show_step_details(step, df, target_name, X, y):
    before_df, after_df = get_step_frames(step, df, target_name, X, y)

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

    fig = create_step_impact_chart(step, df, target_name, before_df, after_df)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")


st.title("Naive Bayes Visualization App")
st.write("Explore the performance of Naive Bayes with interactive visualizations.")

st.sidebar.header("Settings")

sidebar_section("Variance Smoothing")
var_smoothing = st.sidebar.slider(
    "Variance Smoothing",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    label_visibility="collapsed",
)
sidebar_divider()

sidebar_section("Select Dataset")
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Breast Cancer Wisconsin", "Wine"],
    label_visibility="collapsed",
)
sidebar_divider()

sidebar_section("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "txt"],
    label_visibility="collapsed",
)
sidebar_divider()

try:
    selected_target_col = None
    if uploaded_file is not None:
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

    X, y, target_name = prepare_features(df, selected_target_col)
    recommended_model_name, model_justification = recommend_nb_classifier(X)

    sidebar_section("Model Recommendation")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Auto", "BernoulliNB", "MultinomialNB", "GaussianNB"],
        label_visibility="collapsed",
    )
    sidebar_divider()

    selected_model_name = recommended_model_name if model_type == "Auto" else model_type
    model, X_test, y_test, y_pred, accuracy, accuracy_label = train_and_evaluate(
        X,
        y,
        selected_model_name,
        var_smoothing,
    )

    st.success("Model trained successfully!")
    st.write(f"Data source: {data_source}")
    st.write(f"Target column: {target_name}")
    st.write(f"**Recommended Model:** {recommended_model_name}")
    st.write(model_justification)
    if model_type == "Auto":
        st.info(f"Using Auto selection: {selected_model_name}")
    else:
        st.success(f"Model override selected: {selected_model_name}")

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric(accuracy_label, f"{accuracy:.2%}")
    metric_2.metric("Rows Used", f"{len(X):,}")
    metric_3.metric("Features", f"{len(X.columns):,}")

    st.plotly_chart(
        create_preprocessing_diagram(df, target_name, len(X.columns)),
        width="stretch",
    )

    sidebar_section("Pipeline Step Details")
    selected_step = st.sidebar.selectbox(
        "Pipeline Step Details",
        get_preprocessing_steps(),
        label_visibility="collapsed",
    )
    sidebar_divider()

    with st.expander("Detailed Step View", expanded=True):
        show_step_details(selected_step, df, target_name, X, y)

    st.plotly_chart(
        build_gaussian_plot(X, y, target_name, var_smoothing),
        width="stretch",
    )
    st.plotly_chart(build_confusion_matrix_plot(y_test, y_pred), width="stretch")

    st.subheader("Data Preview")
    st.dataframe(df.head(10), width="stretch")

    input_data = build_prediction_input(df, target_name)

    if st.sidebar.button("Predict"):
        try:
            new_data_df = preprocess_prediction_input(input_data, X.columns)
            prediction = model.predict(new_data_df)[0]
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
except Exception as e:
    st.warning(f"Error loading or processing the dataset: {e}")

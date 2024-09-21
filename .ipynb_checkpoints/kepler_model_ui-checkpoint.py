import io
import base64
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import tensorflow as tf
from joblib import dump
from joblib import load
from tensorflow import keras
import matplotlib.pyplot as plt
from shiny import App, ui, render
from xgboost import XGBClassifier
from tensorflow.keras import layers
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
warnings.filterwarnings("ignore")

# Define the UI layout
app_ui = ui.page_navbar(
    ui.nav_panel("Data Input",ui.div(ui.input_file("file_upload", "Upload CSV File"),ui.output_ui("data_input_content"),)),
    ui.nav_panel("Dashboard",ui.div(ui.h4("Exploratory Visualization"),
            ui.input_select("visualization_type", "Choose a visualization",choices=["Class Distribution", "Histograms", "Box Plots", "Correlation Matrix"]),
            ui.output_ui("visualization_output"))),
    ui.nav_panel("Data Modeling",ui.div(ui.h4("Models"),
            ui.input_select("model_choice", "Choose Model", choices=["Logistic Regression","XG Boost","Cat Boost","RNN","GRU"]),
            ui.output_ui("modeling_content"),)),
    ui.nav_panel("Predictions",
        ui.div(
            ui.h4("Model Predictions"),
            ui.layout_columns(
                ui.div(
                    ui.input_text("koi_fpflag_co", "KOI Fpflag CO"),
                    ui.input_text("koi_steff_err2", "KOI Steff Err2"),
                    ui.input_text("koi_prad", "KOI Prad"),
                    ui.input_text("koi_steff_err1", "KOI Steff Err1"),
                    style="flex: 50%; padding: 5px;"
                ),
                ui.div(
                    ui.input_text("koi_fpflag_ss", "KOI Fpflag SS"),
                    ui.input_text("koi_fpflag_nt", "KOI Fpflag NT"),
                    ui.input_text("koi_prad_err1", "KOI Prad Err1"),
                    ui.input_text("koi_prad_err2", "KOI Prad Err2"),
                    style="flex: 50%; padding: 5px;"
                )
            ),
            ui.input_select("model_select", "Choose Model", choices=["Logistic Regression", "XG Boost", "Cat Boost", "RNN", "GRU"]),
            ui.output_ui("prediction_result"))),
    ui.nav_spacer(),
    ui.nav_control(ui.input_dark_mode(mode="dark")),  # Adds dark mode switch to the navbar
    title="Kepler Mission Data Explorer")

# Define server logic
def server(input, output, session):
    kepler = None
    @output
    @render.ui
    def data_input_content():
        nonlocal kepler
        file = input.file_upload()
        if file:
            # Access the path of the first uploaded file
            file_path = file[0]['datapath']
            # Load the CSV file
            kepler = pd.read_csv(file_path)
            # Remove specified columns
            columns_to_remove = ['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score']
            kepler = kepler.drop(columns=columns_to_remove, errors='ignore')
            # Remove columns with more than 80% missing values
            threshold = 0.8 * len(kepler)
            kepler = kepler.dropna(thresh=threshold, axis=1)
            # Impute missing values
            for column in kepler.columns:
                if kepler[column].dtype == 'object':
                    kepler[column].fillna(kepler[column].mode()[0], inplace=True)
                else:
                    kepler[column].fillna(kepler[column].mean(), inplace=True)
            # Drop duplicates
            kepler = kepler.drop_duplicates()
            # Capture the data info using StringIO
            buffer = io.StringIO()
            kepler.info(buf=buffer)
            info_str = buffer.getvalue()
            # Prepare the output summary in HTML format
            summary_html = f"""
            <h4>Displaying the first few rows of the dataset:</h4>
            {kepler.head().to_html(classes='table table-striped', index=False)}
            <h4>Displaying the data information:</h4>
            <pre>{info_str}</pre>
            <h4>Displaying the list of data columns:</h4>
            <pre>{kepler.columns.tolist()}</pre>
            <h4>Displaying the shape of the dataset:</h4>
            <pre>{kepler.shape}</pre>
            <h4>Displaying the statistical description of the dataset:</h4>
            {kepler.describe().to_html(classes='table table-striped', index=False)}
            """
            return ui.HTML(summary_html)
        return "Please upload a CSV file."
    @output
    @render.ui
    def visualization_output():
        nonlocal kepler
        if kepler is None:
            return "No data available. Please upload a CSV file in the Data Input tab."
        vis_type = input.visualization_type()
        if vis_type == "Class Distribution":
            return render_class_distribution(kepler)
        elif vis_type == "Histograms":
            return render_histograms(kepler)
        elif vis_type == "Box Plots":
            return render_box_plots(kepler)
        elif vis_type == "Correlation Matrix":
            return render_correlation_matrix(kepler)
        return "Please select a visualization option."
    def render_class_distribution(kepler):
        kepler = kepler[kepler['koi_disposition'] != 'CANDIDATE']
        confirmed_count = kepler[kepler['koi_disposition'] == 'CONFIRMED'].shape[0]
        false_positive_count = kepler[kepler['koi_disposition'] == 'FALSE POSITIVE'].shape[0]
        counts = [confirmed_count, false_positive_count]
        labels = ['CONFIRMED', 'FALSE POSITIVE']
        plt.figure(figsize=(6, 5))
        plt.bar(labels, counts, color=['blue', 'green'])
        plt.xlabel('Disposition')
        plt.ylabel('Count')
        plt.title('Count of Confirmed and False Positive after Imputation')
        return plot_to_html()
    def render_histograms(kepler):
        kepler.hist(bins=30, edgecolor='black', figsize=(15, 15))
        plt.tight_layout()
        return plot_to_html()
    def render_box_plots(kepler):
        kepler.plot(kind='box', subplots=True, layout=(int(np.ceil(len(kepler.columns) / 4)), 4), figsize=(15, 15), sharex=False, sharey=False)
        plt.tight_layout()
        return plot_to_html()
    def render_correlation_matrix(kepler):
        numeric_kepler = kepler.select_dtypes(include=[np.number])
        corr_matrix = numeric_kepler.corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 6})
        plt.title('Correlation Matrix')
        return plot_to_html()
    @output
    @render.ui
    def modeling_content():
        nonlocal kepler
        if kepler is None:
            return "Please upload data to run the model."
        # Make a copy of the original dataset to ensure preprocessing does not affect the original dataset
        kepler_copy = kepler.copy()
        # Map 'koi_disposition' and immediately drop rows with NaN values
        disposition_mapping = {'FALSE POSITIVE': 0, 'CONFIRMED': 1}
        kepler_copy['koi_disposition'] = kepler_copy['koi_disposition'].map(disposition_mapping)
        kepler_copy.dropna(subset=['koi_disposition'], inplace=True)
        kepler_copy['koi_disposition'] = kepler_copy['koi_disposition'].astype(int)
        if 'koi_tce_delivname' in kepler_copy.columns:
            kepler_copy.drop(['koi_tce_delivname'], axis=1, inplace=True)
        if 'ra_str' in kepler_copy.columns:
            kepler_copy['ra_str'] = kepler_copy['ra_str'].apply(sexagesimal_to_degrees)
            kepler_copy.rename(columns={'ra_str': 'ra_deg'}, inplace=True)
        if 'dec_str' in kepler_copy.columns:
            kepler_copy['dec_str'] = kepler_copy['dec_str'].apply(dec_str_to_degrees)
            kepler_copy.rename(columns={'dec_str': 'dec_deg'}, inplace=True)
        # Setup data for modeling
        X = kepler_copy.drop(['koi_disposition'], axis=1)
        Y = kepler_copy['koi_disposition']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
        np.random.seed(42)
        # Initialize the RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # Fit the model
        rf.fit(X_train, y_train)
        # Get feature importances
        feature_importances = rf.feature_importances_
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({'Feature': X_train.columns,'Importance': feature_importances})
        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        # Select the top 8 features based on importance
        top_8_features = importance_df.head(8)['Feature'].values
        # Subset the training and test sets with these top 8 features
        X_train_top_8 = X_train[top_8_features]
        X_test_top_8 = X_test[top_8_features]
        # Model selection
        model_type = input.model_choice()
        if model_type == "Logistic Regression":
            lr = LogisticRegression()
            lr.fit(X_train_top_8, y_train)
            print(X_train_top_8.columns)
            y_pred_lr = lr.predict(X_test_top_8)
            dump(lr, 'logistic_regression_model.joblib')
            return display_model_results(y_test, y_pred_lr, lr, X_test_top_8)
        elif model_type == "XG Boost":
            xgb = XGBClassifier(random_state=369)
            xgb.fit(X_train_top_8, y_train)
            print(X_train_top_8.columns)
            y_pred_xgb = xgb.predict(X_test_top_8)
            dump(xgb, 'xgboost_model.joblib')
            return display_model_results(y_test, y_pred_xgb, xgb, X_test_top_8)
        elif model_type == "Cat Boost":
            cat = CatBoostClassifier(random_state=369, verbose=0)
            cat.fit(X_train_top_8, y_train)
            print(X_train_top_8.columns)
            y_pred_cat = cat.predict(X_test_top_8)
            dump(cat, 'catboost_model.joblib') 
            return display_model_results(y_test, y_pred_cat, cat, X_test_top_8)
        elif model_type == "RNN":
            # Initialize and train the RNN model
            keras.utils.set_random_seed(0)
            rnn = keras.Sequential([
                layers.SimpleRNN(32, input_shape=(X_train_top_8.shape[1], 1)),
                layers.Dense(10, activation='relu'),
                layers.Dense(2, activation='sigmoid')
            ])
            rnn.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            history_rnn = rnn.fit(X_train_top_8, y_train, epochs=10, validation_split=0.02, verbose=0)
            print(X_train_top_8.columns)
            rnn.save('rnn_model.h5') 
            # Make predictions
            pred_prob_rnn = rnn.predict(X_test_top_8)
            y_pred_rnn = np.argmax(pred_prob_rnn, axis=1)
            # Display results
            return display_model_results(y_test, y_pred_rnn, rnn, X_test_top_8, history=history_rnn)
        elif model_type == "GRU":
            # Initialize and train the GRU model
            gru = keras.Sequential([
                layers.GRU(32, input_shape=(X_train_top_8.shape[1], 1)),
                layers.Dense(10, activation='relu'),
                layers.Dense(2, activation='sigmoid')
            ])
            gru.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            history_gru = gru.fit(X_train_top_8, y_train, epochs=10, validation_split=0.02, verbose=0)
            print(X_train_top_8.columns)
            gru.save('gru_model.h5') 
            # Make predictions
            pred_prob_gru = gru.predict(X_test_top_8)
            y_pred_gru = np.argmax(pred_prob_gru, axis=1)
            # Display results
            return display_model_results(y_test, y_pred_gru, gru, X_test_top_8, history=history_gru)
    def display_model_results(y_test, y_pred, model, X_test, history=None):
        y_test_copy = y_test.copy()
        y_pred_copy = y_pred.copy()
        X_test_copy = X_test.copy()
        # Ensure indices align for sampling
        X_test_sample = X_test_copy.reset_index(drop=True)
        y_test_sample = y_test_copy.reset_index(drop=True)
        y_pred_sample = pd.Series(y_pred_copy).reset_index(drop=True)
        # Sample data for display
        sample_indices = np.random.choice(X_test_sample.index, size=min(10, len(y_test_sample)), replace=False)
        sample_data = X_test_sample.loc[sample_indices]
        sample_actual = y_test_sample[sample_indices]
        sample_predictions = y_pred_sample[sample_indices]
        # Combine samples into a DataFrame
        label_dict = {0: 'FALSE POSITIVE', 1: 'CONFIRMED'}
        sample_df = sample_data.copy()
        sample_df['Actual'] = [label_dict[label] for label in sample_actual]
        sample_df['Predicted'] = [label_dict[label] for label in sample_predictions]
        sample_display_html = sample_df.to_html(classes='table table-striped')
        # Setup figure based on whether history is provided
        if history:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            axs[0].plot(history.history['accuracy'], label='Training Accuracy')
            axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axs[0].set_title('Model Training Accuracy')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Accuracy')
            axs[0].legend(loc='upper left')
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        y_test_decoded = [label_dict[label] for label in y_test]
        y_pred_decoded = [label_dict[label] for label in y_pred]
        # Generate classification report and convert to HTML
        report = classification_report(y_test_decoded, y_pred_decoded, digits=4)
        classification_html = f"<h4>Classification Report:</h4><pre>{report}</pre>"
        # Confusion Matrix
        sns.heatmap(confusion_matrix(y_test_decoded, y_pred_decoded), annot=True, fmt='d', cmap='Blues', ax=axs[-2])
        axs[-2].set_title('Confusion Matrix')
        axs[-2].set_xlabel('Predicted')
        axs[-2].set_ylabel('True')
        # ROC Curve
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Use the second column which represents the probability of the positive class
            y_prob = model.predict(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axs[-1].plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
        axs[-1].plot([0, 1], [0, 1], linestyle='--')
        axs[-1].set_title('ROC Curve')
        axs[-1].set_xlabel('False Positive Rate')
        axs[-1].set_ylabel('True Positive Rate')
        axs[-1].legend(loc='lower right')
        plt.tight_layout()
        plot_html = plot_to_html()
        combined_html = f"<div>{sample_display_html}</div><div>{classification_html}</div><center><div>{plot_html}</div></center>"
        return ui.HTML(combined_html)
    
    def sexagesimal_to_degrees(sexagesimal):
        parts = sexagesimal.split('h')
        hours = float(parts[0])
        parts = parts[1].split('m')
        minutes = float(parts[0])
        parts = parts[1].split('s')
        seconds = float(parts[0])
        degrees = hours * 15 + minutes * 0.25 + seconds * (1/240)
        return degrees
    
    def dec_str_to_degrees(dec_str):
        parts = dec_str.split('d')
        degrees = float(parts[0])
        parts = parts[1].split('m')
        minutes = float(parts[0])
        seconds = float(parts[1].rstrip('s'))
        return degrees + minutes / 60 + seconds / 3600
    
    def plot_to_html():
        """Convert a Matplotlib plot to HTML."""
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return ui.HTML(f"<img src='data:image/png;base64,{data}'/>")
    
    @output
    @render.ui
    def prediction_result():
        # Get input values and check for completeness
        try:
            base_features = {
                'koi_fpflag_co': float(input.koi_fpflag_co()),
                'koi_steff_err2': float(input.koi_steff_err2()),
                'koi_prad': float(input.koi_prad()),
                'koi_steff_err1': float(input.koi_steff_err1()),
                'koi_fpflag_ss': float(input.koi_fpflag_ss()),
                'koi_fpflag_nt': float(input.koi_fpflag_nt()),
                'koi_prad_err1': float(input.koi_prad_err1()),
                'koi_prad_err2': float(input.koi_prad_err2())
            }
        except ValueError as e:
            return ui.HTML("<h4>Please fill out all fields correctly before predicting.</h4>")
        # Load the appropriate model based on user selection
        model_type = input.model_select()
        if model_type == "Logistic Regression":
            model = load_model('logistic_regression_model.joblib')
            feature_order = ['koi_fpflag_co', 'koi_steff_err2', 'koi_prad', 'koi_steff_err1',
                             'koi_fpflag_ss', 'koi_fpflag_nt', 'koi_prad_err1', 'koi_prad_err2']
        elif model_type == "XG Boost":
            model = load_model('xgboost_model.joblib')
            feature_order = ['koi_fpflag_co', 'koi_steff_err2', 'koi_prad', 'koi_steff_err1',
                             'koi_fpflag_ss', 'koi_fpflag_nt', 'koi_prad_err1', 'koi_prad_err2']
        elif model_type == "Cat Boost":
            model = load_model('catboost_model.joblib')
            feature_order = ['koi_fpflag_co', 'koi_steff_err2', 'koi_prad', 'koi_steff_err1',
                             'koi_fpflag_ss', 'koi_fpflag_nt', 'koi_prad_err1', 'koi_prad_err2']
        elif model_type == "RNN" or model_type == "GRU":
            model = keras.models.load_model(f'{model_type.lower()}_model.h5')
            feature_order = ['koi_fpflag_co', 'koi_steff_err2', 'koi_prad', 'koi_steff_err1',
                             'koi_fpflag_ss', 'koi_fpflag_nt', 'koi_prad_err1', 'koi_prad_err2']
        # Reorder features as per model requirement
        ordered_features = {key: base_features[key] for key in feature_order}
        features_df = pd.DataFrame([ordered_features])
        # Predict using the loaded model
        prediction = model.predict(features_df)
        prediction = np.argmax(prediction) if model_type in ["RNN", "GRU"] else prediction[0]
        result = "Confirmed" if prediction == 1 else "False Positive"
        return ui.HTML(f"<h4>Prediction: {result}</h4>")
    
    def load_model(filename):
        return load(filename)
    
app = App(app_ui, server)
# Run the application
if __name__ == "__main__":
    app.run(port=8000)
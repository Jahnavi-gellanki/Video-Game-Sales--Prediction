# # # # # # from flask import Flask, render_template, request
# # # # # # import pandas as pd
# # # # # # import joblib
# # # # # # import os
# # # # # # import gdown

# # # # # # app = Flask(__name__)

# # # # # # # -----------------------------
# # # # # # # Load model (from local or Google Drive)
# # # # # # # -----------------------------
# # # # # # model_path = 'video_game_sales_best_model.pkl'

# # # # # # # If model not present locally, download from Google Drive
# # # # # # gdrive_url = 'https://drive.google.com/file/d/1YZ76NOdRVD60DHdKDZfgIeGpu4G4iFGW/view?usp=drive_link'  # replace with your model file ID
# # # # # # if not os.path.exists(model_path):
# # # # # #     print("Downloading model from Google Drive...")
# # # # # #     gdown.download(gdrive_url, model_path, quiet=False)

# # # # # # # Load the pipeline/model
# # # # # # model = joblib.load(model_path)
# # # # # # print("Model loaded successfully!")

# # # # # # # -----------------------------
# # # # # # # Home page
# # # # # # # -----------------------------
# # # # # # @app.route('/')
# # # # # # def home():
# # # # # #     return render_template('index.html')

# # # # # # # -----------------------------
# # # # # # # Predict route
# # # # # # # -----------------------------
# # # # # # @app.route('/predict', methods=['POST'])
# # # # # # def predict():
# # # # # #     # Get input from form
# # # # # #     platform = request.form['platform']
# # # # # #     year = int(request.form['year'])
# # # # # #     genre = request.form['genre']
# # # # # #     publisher = request.form['publisher']

# # # # # #     # Prepare input for model
# # # # # #     input_df = pd.DataFrame([{
# # # # # #         'Platform': platform,
# # # # # #         'Year': year,
# # # # # #         'Genre': genre,
# # # # # #         'Publisher': publisher
# # # # # #     }])

# # # # # #     # Predict regional sales
# # # # # #     pred = model.predict(input_df)[0]
# # # # # #     na, eu, jp, other = pred
# # # # # #     global_sales = na + eu + jp + other

# # # # # #     # Pass results back to frontend
# # # # # #     return render_template(
# # # # # #         'index.html',
# # # # # #         prediction=True,
# # # # # #         na=round(na,3),
# # # # # #         eu=round(eu,3),
# # # # # #         jp=round(jp,3),
# # # # # #         other=round(other,3),
# # # # # #         global_sales=round(global_sales,3)
# # # # # #     )

# # # # # # # -----------------------------
# # # # # # # Run app
# # # # # # # -----------------------------
# # # # # # if __name__ == '__main__':
# # # # # #     app.run(debug=True)

# # # # # from flask import Flask, render_template, request
# # # # # import pandas as pd
# # # # # import joblib
# # # # # import os
# # # # # import gdown

# # # # # app = Flask(__name__)

# # # # # # -----------------------------
# # # # # # Load model (from local or Google Drive)
# # # # # # -----------------------------
# # # # # model_path = 'video_game_sales_best_model.pkl'

# # # # # # If model not present locally, download from Google Drive
# # # # # gdrive_url = 'https://drive.google.com/file/d/1YZ76NOdRVD60DHdKDZfgIeGpu4G4iFGW/view?usp=drive_link'  # direct download link format
# # # # # if not os.path.exists(model_path):
# # # # #     print("Downloading model from Google Drive...")
# # # # #     gdown.download(gdrive_url, model_path, quiet=False)

# # # # # # Load the pipeline/model
# # # # # model = joblib.load(model_path)
# # # # # print("✅ Model loaded successfully!")

# # # # # # -----------------------------
# # # # # # Home page
# # # # # # -----------------------------
# # # # # @app.route('/')
# # # # # def home():
# # # # #     return render_template('index.html')

# # # # # # -----------------------------
# # # # # # Predict route
# # # # # # -----------------------------
# # # # # @app.route('/predict', methods=['POST'])
# # # # # def predict():
# # # # #     try:
# # # # #         # Get input from form
# # # # #         platform = request.form['platform']
# # # # #         year = int(request.form['year'])
# # # # #         genre = request.form['genre']
# # # # #         publisher = request.form['publisher']

# # # # #         # Prepare input for model
# # # # #         input_df = pd.DataFrame([{
# # # # #             'Platform': platform,
# # # # #             'Year': year,
# # # # #             'Genre': genre,
# # # # #             'Publisher': publisher
# # # # #         }])

# # # # #         # Add derived features (these were used during training)
# # # # #         input_df['Decade'] = (input_df['Year'] // 10) * 10  # e.g. 2006 → 2000
# # # # #         input_df['Publisher_short'] = input_df['Publisher'].str[:3]  # e.g. 'Nintendo' → 'Nin'

# # # # #         # Predict regional sales
# # # # #         pred = model.predict(input_df)[0]
# # # # #         na, eu, jp, other = pred
# # # # #         global_sales = na + eu + jp + other

# # # # #         # Pass results back to frontend
# # # # #         return render_template(
# # # # #             'index.html',
# # # # #             prediction=True,
# # # # #             na=round(na, 3),
# # # # #             eu=round(eu, 3),
# # # # #             jp=round(jp, 3),
# # # # #             other=round(other, 3),
# # # # #             global_sales=round(global_sales, 3)
# # # # #         )

# # # # #     except Exception as e:
# # # # #         # Handle any runtime errors gracefully
# # # # #         return render_template('index.html', prediction=False, error=str(e))

# # # # # # -----------------------------
# # # # # # Run app
# # # # # # -----------------------------
# # # # # if __name__ == '__main__':
# # # # #     app.run(debug=True)


# # # # from flask import Flask, render_template, request
# # # # import pandas as pd
# # # # import joblib
# # # # import os
# # # # import gdown

# # # # app = Flask(__name__)

# # # # # -----------------------------
# # # # # Load model (from local or Google Drive)
# # # # # -----------------------------
# # # # model_path = 'video_game_sales_best_model.pkl'

# # # # # Google Drive link (convert to direct-download format)
# # # # gdrive_url = 'https://drive.google.com/file/d/1YZ76NOdRVD60DHdKDZfgIeGpu4G4iFGW/view?usp=drive_link'

# # # # # Download if missing
# # # # if not os.path.exists(model_path):
# # # #     print("Downloading model from Google Drive...")
# # # #     gdown.download(gdrive_url, model_path, quiet=False)

# # # # # Load model
# # # # model = joblib.load(model_path)
# # # # print("✅ Model loaded successfully!")

# # # # # -----------------------------
# # # # # Home page
# # # # # -----------------------------
# # # # @app.route('/')
# # # # def home():
# # # #     return render_template('index.html')

# # # # # -----------------------------
# # # # # Predict route
# # # # # -----------------------------
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         # Get input from form
# # # #         platform = request.form['platform']
# # # #         year = int(request.form['year'])
# # # #         genre = request.form['genre']
# # # #         publisher = request.form['publisher']

# # # #         # Prepare DataFrame
# # # #         input_df = pd.DataFrame([{
# # # #             'Platform': platform,
# # # #             'Year': year,
# # # #             'Genre': genre,
# # # #             'Publisher': publisher
# # # #         }])

# # # #         # ✅ Add derived columns used during training
# # # #         input_df['Decade'] = (input_df['Year'] // 10) * 10
# # # #         input_df['Publisher_short'] = input_df['Publisher'].str[:3]

# # # #         print("\n🔹 Input DataFrame for Prediction:")
# # # #         print(input_df)

# # # #         # Make prediction
# # # #         pred = model.predict(input_df)

# # # #         print("🔹 Raw prediction output:", pred)

# # # #         # Some models return array([[na, eu, jp, other]])
# # # #         # Ensure shape compatibility
# # # #         if isinstance(pred, (list, tuple)) or pred.ndim > 1:
# # # #             na, eu, jp, other = pred[0]
# # # #         else:
# # # #             # If only one value is returned
# # # #             na = eu = jp = other = pred[0]

# # # #         global_sales = na + eu + jp + other

# # # #         return render_template(
# # # #             'index.html',
# # # #             prediction=True,
# # # #             na=round(na, 3),
# # # #             eu=round(eu, 3),
# # # #             jp=round(jp, 3),
# # # #             other=round(other, 3),
# # # #             global_sales=round(global_sales, 3)
# # # #         )

# # # #     except Exception as e:
# # # #         print("❌ Error during prediction:", e)
# # # #         return render_template('index.html', error=f"Prediction failed: {e}")

# # # # # -----------------------------
# # # # # Run Flask App
# # # # # -----------------------------
# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)


# # # from flask import Flask, render_template, request
# # # import pandas as pd
# # # import joblib
# # # import os
# # # import gdown
# # # import numpy as np

# # # app = Flask(__name__)

# # # # -----------------------------
# # # # Load model (from local or Google Drive)
# # # # -----------------------------
# # # model_path = 'video_game_sales_best_model.pkl'
# # # gdrive_url = 'https://drive.google.com/file/d/1YZ76NOdRVD60DHdKDZfgIeGpu4G4iFGW/view?usp=drive_link'

# # # if not os.path.exists(model_path):
# # #     print("Downloading model from Google Drive...")
# # #     gdown.download(gdrive_url, model_path, quiet=False)

# # # model = joblib.load(model_path)
# # # print("✅ Model loaded successfully!")

# # # # -----------------------------
# # # # Home page
# # # # -----------------------------
# # # @app.route('/')
# # # def home():
# # #     return render_template('index.html')

# # # # -----------------------------
# # # # Predict route
# # # # -----------------------------
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         # Get inputs from form
# # #         platform = request.form['platform']
# # #         year = int(request.form['year'])
# # #         genre = request.form['genre']
# # #         publisher = request.form['publisher']

# # #         # Prepare dataframe
# # #         input_df = pd.DataFrame([{
# # #             'Platform': platform,
# # #             'Year': year,
# # #             'Genre': genre,
# # #             'Publisher': publisher
# # #         }])

# # #         # Derived features (used during model training)
# # #         input_df['Decade'] = (input_df['Year'] // 10) * 10
# # #         input_df['Publisher_short'] = input_df['Publisher'].str[:3]

# # #         print("\n🔹 Input DataFrame for Prediction:")
# # #         print(input_df)

# # #         # Make prediction
# # #         pred = model.predict(input_df)
# # #         pred = np.array(pred).flatten()  # convert to 1D list

# # #         print("🔹 Raw model output:", pred)

# # #         # Handle both single and multi-output models
# # #         if len(pred) == 4:
# # #             na, eu, jp, other = pred
# # #             global_sales = na + eu + jp + other
# # #         else:
# # #             # Model predicts only global sales
# # #             na = eu = jp = other = 0
# # #             global_sales = pred[0]

# # #         # Return to template
# # #         return render_template(
# # #             'index.html',
# # #             prediction=True,
# # #             na=round(float(na), 3),
# # #             eu=round(float(eu), 3),
# # #             jp=round(float(jp), 3),
# # #             other=round(float(other), 3),
# # #             global_sales=round(float(global_sales), 3)
# # #         )

# # #     except Exception as e:
# # #         print("❌ Error during prediction:", e)
# # #         return render_template('index.html', error=f"Prediction failed: {e}")

# # # # -----------------------------
# # # # Run app
# # # # -----------------------------
# # # if __name__ == '__main__':
# # #     app.run(debug=True)

# # from flask import Flask, render_template, request
# # import pandas as pd
# # import joblib
# # import os
# # import gdown
# # import numpy as np

# # app = Flask(__name__)

# # # -----------------------------
# # # Load model (from local or Google Drive)
# # # -----------------------------
# # model_path = 'video_game_sales_best_model.pkl'
# # gdrive_url = 'https://drive.google.com/file/d/1YZ76NOdRVD60DHdKDZfgIeGpu4G4iFGW/view?usp=drive_link'


# # if not os.path.exists(model_path):
# #     print("Downloading model from Google Drive...")
# #     gdown.download(gdrive_url, model_path, quiet=False)

# # model = joblib.load(model_path)
# # print("✅ Model loaded successfully!")

# # # -----------------------------
# # # Top publishers (same as training)
# # # -----------------------------
# # top_publishers = [
# #     'Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment',
# #     'Ubisoft', 'Take-Two Interactive', 'THQ', 'Konami Digital Entertainment',
# #     'Sega', 'Namco Bandai Games', 'Capcom', 'Square Enix', 'Atari',
# #     'Warner Bros. Interactive Entertainment', 'Microsoft Game Studios',
# #     'LucasArts', 'Bethesda Softworks', 'Eidos Interactive', 'Hudson Soft',
# #     'Midway Games', 'Acclaim Entertainment', '505 Games', 'Deep Silver',
# #     'Codemasters', 'Tecmo Koei', 'Disney Interactive Studios', 'MTV Games',
# #     'Virgin Interactive', 'Infogrames', 'Vivendi Games', 'Other'
# # ]

# # # -----------------------------
# # # Home page
# # # -----------------------------
# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # # -----------------------------
# # # Predict route
# # # -----------------------------
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # Get inputs
# #         platform = request.form['platform'].strip()
# #         year = int(request.form['year'])
# #         genre = request.form['genre'].strip()
# #         publisher = request.form['publisher'].strip()

# #         # 🧠 Match feature engineering from training
# #         publisher_short = publisher if publisher in top_publishers else 'Other'
# #         decade = (year // 10) * 10
# #         platform_genre = f"{platform}_{genre}"

# #         # ✅ Create DataFrame with *exact* same columns as training
# #         input_df = pd.DataFrame([{
# #             'Platform': platform,
# #             'Year': year,
# #             'Genre': genre,
# #             'Publisher_short': publisher_short,
# #             'Decade': decade,
# #             'Platform_Genre': platform_genre
# #         }])

# #         print("\n🔹 Input DataFrame for Prediction:")
# #         print(input_df)

# #         # Make prediction
# #         pred = model.predict(input_df)
# #         pred = np.array(pred).flatten()

# #         print("🔹 Raw model output:", pred)

# #         # Handle 4-output regression (NA, EU, JP, Other)
# #         if len(pred) == 4:
# #             na, eu, jp, other = pred
# #             global_sales = na + eu + jp + other
# #         else:
# #             na = eu = jp = other = 0
# #             global_sales = pred[0]

# #         return render_template(
# #             'index.html',
# #             prediction=True,
# #             na=round(float(na), 3),
# #             eu=round(float(eu), 3),
# #             jp=round(float(jp), 3),
# #             other=round(float(other), 3),
# #             global_sales=round(float(global_sales), 3)
# #         )

# #     except Exception as e:
# #         print("❌ Error during prediction:", e)
# #         return render_template('index.html', error=f"Prediction failed: {e}")

# # # -----------------------------
# # # Run app
# # # -----------------------------
# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# import os
# import gdown
# import numpy as np

# app = Flask(__name__)

# # -----------------------------
# # Model configuration
# # -----------------------------
# MODEL_PATH = 'video_game_sales_best_model.pkl'
# # Use the 'uc?id=FILE_ID' format for Google Drive
# GDRIVE_URL = 'https://drive.google.com/uc?id=1YZ76NOdRVD60DHdKDZfgIeGpu4G4iFGW'

# def download_model():
#     """Download the model from Google Drive if not already present."""
#     if not os.path.exists(MODEL_PATH):
#         print("Downloading model from Google Drive...")
#         try:
#             gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
#             print("✅ Model downloaded successfully!")
#         except Exception as e:
#             print(f"❌ Failed to download model: {e}")

# # Download the model if needed
# download_model()

# # Load the model
# try:
#     model = joblib.load(MODEL_PATH)
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Failed to load model: {e}")
#     model = None  # Fallback in case of errors

# # -----------------------------
# # Top publishers (from training)
# # -----------------------------
# top_publishers = [
#     'Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment',
#     'Ubisoft', 'Take-Two Interactive', 'THQ', 'Konami Digital Entertainment',
#     'Sega', 'Namco Bandai Games', 'Capcom', 'Square Enix', 'Atari',
#     'Warner Bros. Interactive Entertainment', 'Microsoft Game Studios',
#     'LucasArts', 'Bethesda Softworks', 'Eidos Interactive', 'Hudson Soft',
#     'Midway Games', 'Acclaim Entertainment', '505 Games', 'Deep Silver',
#     'Codemasters', 'Tecmo Koei', 'Disney Interactive Studios', 'MTV Games',
#     'Virgin Interactive', 'Infogrames', 'Vivendi Games', 'Other'
# ]

# # -----------------------------
# # Home page
# # -----------------------------
# @app.route('/')
# def home():
#     return render_template('index.html')

# # -----------------------------
# # Predict route
# # -----------------------------
# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return render_template('index.html', error="Model not loaded. Prediction unavailable.")

#     try:
#         # Get inputs
#         platform = request.form['platform'].strip()
#         year = int(request.form['year'])
#         genre = request.form['genre'].strip()
#         publisher = request.form['publisher'].strip()

#         # Feature engineering
#         publisher_short = publisher if publisher in top_publishers else 'Other'
#         decade = (year // 10) * 10
#         platform_genre = f"{platform}_{genre}"

#         # Prepare input DataFrame
#         input_df = pd.DataFrame([{
#             'Platform': platform,
#             'Year': year,
#             'Genre': genre,
#             'Publisher_short': publisher_short,
#             'Decade': decade,
#             'Platform_Genre': platform_genre
#         }])

#         print("\n🔹 Input DataFrame for Prediction:")
#         print(input_df)

#         # Make prediction
#         pred = model.predict(input_df)
#         pred = np.array(pred).flatten()
#         print("🔹 Raw model output:", pred)

#         # Handle 4-output regression (NA, EU, JP, Other)
#         if len(pred) == 4:
#             na, eu, jp, other = pred
#             global_sales = na + eu + jp + other
#         else:
#             na = eu = jp = other = 0
#             global_sales = pred[0]

#         return render_template(
#             'index.html',
#             prediction=True,
#             na=round(float(na), 3),
#             eu=round(float(eu), 3),
#             jp=round(float(jp), 3),
#             other=round(float(other), 3),
#             global_sales=round(float(global_sales), 3)
#         )

#     except Exception as e:
#         print("❌ Error during prediction:", e)
#         return render_template('index.html', error=f"Prediction failed: {e}")

# # -----------------------------
# # Run app
# # -----------------------------
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import gdown
import numpy as np

app = Flask(__name__)

# -----------------------------
# Model configuration
# -----------------------------
MODEL_PATH = 'video_game_sales_best_model.pkl'
# Use the 'uc?id=FILE_ID' format for Google Drive
GDRIVE_URL = 'https://drive.google.com/uc?id=1YZ76NOdRVD60DHdKDZfgIeGpu4G4iFGW'

def download_model():
    """Download the model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")

# Download the model if needed
download_model()

# -----------------------------
# Load model with compatibility fix
# -----------------------------
try:
    # Temporary fix for sklearn internal object mismatch
    import sklearn.compose._column_transformer

    class RemainderColList:
        pass

    # Attach it so joblib can find it
    sklearn.compose._column_transformer.RemainderColList = RemainderColList

    # Now load the model
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None  # Fallback in case of errors

# -----------------------------
# Top publishers (from training)
# -----------------------------
top_publishers = [
    'Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment',
    'Ubisoft', 'Take-Two Interactive', 'THQ', 'Konami Digital Entertainment',
    'Sega', 'Namco Bandai Games', 'Capcom', 'Square Enix', 'Atari',
    'Warner Bros. Interactive Entertainment', 'Microsoft Game Studios',
    'LucasArts', 'Bethesda Softworks', 'Eidos Interactive', 'Hudson Soft',
    'Midway Games', 'Acclaim Entertainment', '505 Games', 'Deep Silver',
    'Codemasters', 'Tecmo Koei', 'Disney Interactive Studios', 'MTV Games',
    'Virgin Interactive', 'Infogrames', 'Vivendi Games', 'Other'
]

# -----------------------------
# Home page
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# Predict route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model not loaded. Prediction unavailable.")

    try:
        # Get inputs
        platform = request.form['platform'].strip()
        year = int(request.form['year'])
        genre = request.form['genre'].strip()
        publisher = request.form['publisher'].strip()

        # Feature engineering
        publisher_short = publisher if publisher in top_publishers else 'Other'
        decade = (year // 10) * 10
        platform_genre = f"{platform}_{genre}"

        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            'Platform': platform,
            'Year': year,
            'Genre': genre,
            'Publisher_short': publisher_short,
            'Decade': decade,
            'Platform_Genre': platform_genre
        }])

        print("\n🔹 Input DataFrame for Prediction:")
        print(input_df)

        # Make prediction
        pred = model.predict(input_df)
        pred = np.array(pred).flatten()
        print("🔹 Raw model output:", pred)

        # Handle 4-output regression (NA, EU, JP, Other)
        if len(pred) == 4:
            na, eu, jp, other = pred
            global_sales = na + eu + jp + other
        else:
            na = eu = jp = other = 0
            global_sales = pred[0]

        return render_template(
            'index.html',
            prediction=True,
            na=round(float(na), 3),
            eu=round(float(eu), 3),
            jp=round(float(jp), 3),
            other=round(float(other), 3),
            global_sales=round(float(global_sales), 3)
        )

    except Exception as e:
        print("❌ Error during prediction:", e)
        return render_template('index.html', error=f"Prediction failed: {e}")

# -----------------------------
# Run app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)

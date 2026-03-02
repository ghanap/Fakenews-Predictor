Fakenews-Predictor

<div class="section">
	<h2>Functional Specifications</h2>
	<h3>Strategy 1: Streamlit Dual-Model (app (1).py)</h3>
	<ul>
		<li><strong>Input:</strong> User submits a news article or headline via a Streamlit web interface, selects a sample, or uploads a CSV for batch analysis.</li>
		<li><strong>Processing:</strong> Text is cleaned, lemmatized, and embedded using SentenceBERT (MiniLM-L6-v2).</li>
		<li><strong>Prediction:</strong> Two models (MLP neural network and SGD linear classifier) predict and display verdicts (Fake/Real) with confidence scores.</li>
		<li><strong>Explainability:</strong> LIME and SHAP visualizations show word-level and embedding-dimension influences. Confusion matrix and calibration plots are available for uploaded test results.</li>
		<li><strong>Model Training:</strong> Models are trained on SBERT embeddings and saved as <code>mlp_classifier_model.pkl</code> and <code>sgd_classifier_model.pkl</code>. Retraining is supported using new labeled data (e.g., <code>train.csv</code>).</li>
	</ul>
	<h3>Strategy 2: Flask/Vectorizer (app.py, fake_news_app.py)</h3>
	<ul>
		<li><strong>Input:</strong> User submits a news article or headline via a Flask web interface or API.</li>
		<li><strong>Processing:</strong> Text is cleaned and vectorized using a traditional method (e.g., TF-IDF, <code>vectorizer.joblib</code>).</li>
		<li><strong>Prediction:</strong> A single machine learning model (e.g., Logistic Regression, Naive Bayes) predicts Fake/Real.</li>
		<li><strong>Model Training:</strong> Model is trained on vectorized text and saved as <code>model.joblib</code>. Retraining is supported using new labeled data (e.g., <code>train.csv</code>).</li>
	</ul>
</div>

<div class="section">
	<h2>Technical Workflow</h2>
	<h3>Strategy 1: Streamlit Dual-Model</h3>
	<ol>
		<li><strong>Data Ingestion:</strong> Load training data from <code>train.csv</code>.</li>
		<li><strong>Preprocessing:</strong> Clean, remove stopwords, and lemmatize text.</li>
		<li><strong>Embedding:</strong> Use SentenceBERT (MiniLM-L6-v2) to convert text to 384-dimensional dense vectors.</li>
		<li><strong>Model Training:</strong> Train two classifiers on embeddings:
			<ul>
				<li>MLP (Multi-Layer Perceptron) for non-linear decision boundaries</li>
				<li>SGD (Stochastic Gradient Descent, linear/SVM-like) for fast, interpretable baseline</li>
			</ul>
			Save models as <code>mlp_classifier_model.pkl</code> and <code>sgd_classifier_model.pkl</code>.
		</li>
		<li><strong>Prediction:</strong> For new input, preprocess and embed text, then predict with both models. Display verdicts and confidence scores.</li>
		<li><strong>Explainability:</strong> Use LIME for word-level impact and SHAP for embedding-dimension importance. Show confusion matrix and calibration for batch results.</li>
		<li><strong>Web App:</strong> Streamlit interface (app (1).py) for interactive analysis, model comparison, and explainability.</li>
	</ol>
	<h3>Strategy 2: Flask/Vectorizer</h3>
	<ol>
		<li><strong>Data Ingestion:</strong> Load training data from <code>train.csv</code>.</li>
		<li><strong>Preprocessing:</strong> Clean and vectorize text using <code>vectorizer.joblib</code> (e.g., TF-IDF).</li>
		<li><strong>Model Training:</strong> Train a classifier (e.g., Logistic Regression, Naive Bayes) and save as <code>model.joblib</code>.</li>
		<li><strong>Prediction:</strong> For new input, vectorize text and use the trained model to predict label.</li>
		<li><strong>API/Web App:</strong> Serve predictions via Flask app (<code>app.py</code> or <code>fake_news_app.py</code>).</li>
		<li><strong>Batch Processing:</strong> Accept CSV uploads, process each row, and return results.</li>
	</ol>
</div>

<div class="section tech-stack">
	<h2>Technology Stack</h2>
	<h3>Strategy 1: Streamlit Dual-Model</h3>
	<ul>
		<li><strong>Programming Language:</strong> Python 3.x</li>
		<li><strong>Web Framework:</strong> Streamlit</li>
		<li><strong>Machine Learning:</strong> scikit-learn (MLP, SGD), sentence-transformers (SBERT)</li>
		<li><strong>Explainability:</strong> LIME, SHAP</li>
		<li><strong>Data Handling:</strong> pandas, numpy</li>
		<li><strong>Serialization:</strong> joblib</li>
		<li><strong>Frontend:</strong> Streamlit, HTML/CSS (custom styling)</li>
		<li><strong>Deployment:</strong> Local server, can be containerized (Docker)</li>
	</ul>
	<h3>Strategy 2: Flask/Vectorizer</h3>
	<ul>
		<li><strong>Programming Language:</strong> Python 3.x</li>
		<li><strong>Web Framework:</strong> Flask</li>
		<li><strong>Machine Learning:</strong> scikit-learn (Logistic Regression, Naive Bayes, etc.)</li>
		<li><strong>Vectorization:</strong> TF-IDF, CountVectorizer (<code>vectorizer.joblib</code>)</li>
		<li><strong>Data Handling:</strong> pandas, numpy</li>
		<li><strong>Serialization:</strong> joblib</li>
		<li><strong>Frontend:</strong> HTML/CSS (Flask templates)</li>
		<li><strong>Deployment:</strong> Local server, can be containerized (Docker)</li>
	</ul>
</div>

<div class="section">
	<h2>File Structure Overview</h2>
	<ul>
		<li><code>app (1).py</code>: Streamlit web app (dual-model, explainability, advanced UI)</li>
		<li><code>mlp_classifier_model.pkl</code>, <code>sgd_classifier_model.pkl</code>: Trained model files for Streamlit app</li>
		<li><code>app.py</code>, <code>fake_news_app.py</code>: Flask/vectorizer-based application files</li>
		<li><code>model.joblib</code>, <code>vectorizer.joblib</code>: Model/vectorizer files for Flask strategy</li>
		<li><code>fakenews.ipynb</code>: Jupyter notebook for exploration/training</li>
		<li><code>train.csv</code>: Training data</li>
		<li><code>requirements.txt</code>: Python dependencies</li>
		<li><code>README.md</code>: Project documentation</li>
	</ul>
</div>

<div class="section">
	<h2>How to Run</h2>
	<h3>Strategy 1: Streamlit Dual-Model</h3>
	<ol>
		<li>Install dependencies: <code>pip install -r requirements.txt</code></li>
		<li>Start the Streamlit app: <code>streamlit run "app (1).py"</code></li>
		<li>Access the web interface at <code>http://localhost:8501</code></li>
	</ol>
	<h3>Strategy 2: Flask/Vectorizer</h3>
	<ol>
		<li>Install dependencies: <code>pip install -r requirements.txt</code></li>
		<li>Start the Flask app: <code>python app.py</code> or <code>python fake_news_app.py</code></li>
		<li>Access the web interface at <code>http://localhost:5000</code></li>
	</ol>
</div>

<div class="section">
	<h2>References</h2>
	<ul>
		<li><a href="https://scikit-learn.org/">scikit-learn documentation</a></li>
		<li><a href="https://flask.palletsprojects.com/">Flask documentation</a></li>
	</ul>
</div>

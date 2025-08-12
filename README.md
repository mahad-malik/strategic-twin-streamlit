![Application Screenshot](assets/screenshot.png)
An interactive decision intelligence platform built with Streamlit and Google Gemini. This application serves as a "Strategic Twin" for a fictional manufacturing company, allowing leaders to simulate complex business scenarios, analyze risks, and make data-driven decisions in minutes.

1. Introduction: A New Way to Navigate Complexity
Our industry is facing unprecedented challenges from volatile energy prices, supply chain disruptions, and complex regulations like carbon pricing. To date, understanding how an external event impacts specific product margins has been a slow, manual process involving disconnected spreadsheets and expert guesswork.
To solve this, we have developed the STK Strategic Twin, a dynamic, interactive digital model of our entire business ecosystem. This tool is not just another dashboard; it is a decision infrastructure. Its purpose is to connect external factors directly to our internal KPIs, allowing us to simulate future scenarios, anticipate risks, and make faster, smarter, data-driven decisions with confidence.

2. Key Features
Intelligent Command Bar: Use natural language to drive the simulation. Simply type requests like "Show me 50 units from the German supplier" to instantly see the impact.
Dynamic KPI Dashboards: The main dashboard gives an immediate, real-time pulse of our most critical KPIs: Gross Margin, Total Cost, and Carbon Footprint.
AI-Powered Strategic Briefings: For any scenario, the application uses Google Gemini to generate a narrative summary that includes a key Observation, its business Implication, and a forward-looking Strategic Question. This translates raw data into actionable advice.
Proactive Risk Analysis:
Tariff Analysis: Run parallel simulations to see how increasing tariffs on a supplier erodes margins, identifying the financial "tipping point."
Sensitivity Analysis: A "Tornado Chart" instantly visualizes which business variables (e.g., Energy Costs, Material Costs) pose the biggest threat to profitability.
Purpose-Driven Visualizations: The interface is built with interactive Plotly charts, including cost flow waterfalls, supply route maps, and an ontology graph that makes the entire business ecosystem transparent to the user.
Ontology-Driven Data Model: The application is powered by a networkx directed graph, creating a "Living Ontology" of the business. This flexible structure represents all entities (suppliers, products, materials) and their complex interdependencies.

3. Technology Stack
Frontend & Application Framework: Streamlit
Data Structure & Modeling: NetworkX
Data Manipulation: Pandas
Data Visualization: Plotly
Natural Language & AI Analysis: Google Gemini API

4. Setup & Installation
Follow these steps to run the application locally.
Prerequisites:
Python 3.8+
An active Google AI API Key with the Gemini API enabled.

Step 1: Clone the repository

git clone https://github.com/<Your-GitHub-Username>/<Your-Repository-Name>.git
cd <Your-Repository-Name>
Step 2: Create and activate a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate

Step 3: Install the required libraries
This project includes a requirements.txt file to install all necessary dependencies.
pip install -r requirements.txt

Step 4: Set up your API Key
IMPORTANT: For security, do not paste your API key directly into app.py. Use Streamlit's secrets management.
Create a folder in your project directory named .streamlit.
Inside that folder, create a file named secrets.toml.

Add your API key to this file like this:
GEMINI_API_KEY = "AIzaSy...your...key...here..."
In app.py, change the line where you configure the API key from this:

genai.configure(api_key="AIzaSyBtEvYXwR4A5JumqXM5iciPl_cishyhWUQ")
to this:
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

Step 5: Run the Streamlit Application

streamlit run app.py

Your browser should open a new tab with the application running.

Step 2: Create a requirements.txt File
This is the final preparation step. This file lists all the libraries your project needs so that others can easily install them.
Make sure your venv is active.
Run the following command in your terminal:

pip freeze > requirements.txt
Your folder should now look like this:
venv/
app.py
.gitignore
README.md
requirements.txt

#  STK Simulation Engine

A simulation platform designed for STK Production GmbH to help business users model, simulate, and analyze scenarios, all from natural language input.

---

##  Project Overview

This prototype was developed as part of a technical assessment by Circonomit in July 2025. It handles four major tasks:

1. **Extend the Data Model for Simulations**
2. **Execution & Caching Strategy**
3. **Natural Language to Simulation Model**
4. **Product Thinking & Usability**


##  Features

- 🧾 Natural language to data model mapping (via spaCy & regex)
- 📊 Simulation engine with override support
- ⚙️ Thread-safe parallel attribute evaluation
- 🧠 Gemini-powered summaries
- 🖥️ Streamlit interface for end users

---

##  How to Run

### 1. Install dependencies

pip install -r requirements.txt

 2. Launch the app 

streamlit run src/app.py

📈 Example Inputs
"Electricity price has increased to €0.45 per kWh, and our consumption is now 12,000 kWh monthly."

System will extract attributes, simulate costs, and generate insights.




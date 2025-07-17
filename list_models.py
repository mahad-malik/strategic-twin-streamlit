import google.generativeai as genai

genai.configure(api_key="AIzaSyClCQXNPai8tbT8rM3L6tVnC-WYF69PE2g")

models = list(genai.list_models())
for model in models:
    print(model.name)

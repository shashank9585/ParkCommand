import requests
from duckduckgo_search import DDGS

API_CONFIG = {
    "base_url": "https://apifreellm.com/api/v1/chat",
    "api_key": "apf_qwy2n598j33z8p14ri8omuph",
    "model": "apifreellm",
    "timeout": 50 
}

def call_llm(prompt, system_prompt=None):
    headers = {"Authorization": f"Bearer {API_CONFIG['api_key']}", "Content-Type": "application/json"}
    
    # FIX: The API expects a single "message" string, not an OpenAI-style "messages" array.
    # We combine the system instructions and the user prompt into one clean string.
    if system_prompt:
        full_message = f"System Instructions: {system_prompt}\n\nUser Query: {prompt}"
    else:
        full_message = prompt
        
    payload = {
        "message": full_message,
        "model": API_CONFIG["model"]
    }
    
    try:
        response = requests.post(API_CONFIG["base_url"], headers=headers, json=payload, timeout=API_CONFIG["timeout"])
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response generated.") if data.get("success") else "API Error."
    except Exception as e:
        return f"API Error: {str(e)}."

def scrape_web_context(location):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{location} Bengaluru parking traffic problem", max_results=3))
            web_context = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            return web_context if web_context and len(web_context) > 50 else "NO LIVE DATA FOUND."
    except:
        return "NO LIVE DATA FOUND."
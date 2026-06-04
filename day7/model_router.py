import os, requests, anthropic
from dotenv import load_dotenv

os.chdir(r"F:\finance-ai-course")
load_dotenv()

claude_client = anthropic.Anthropic()
OLLAMA_URL = "http://localhost:11434/api/generate"

def call_claude(prompt):
    r = claude_client.messages.create(
        model="claude-opus-4-5", max_tokens=512, temperature=0.1,
        system="You are a finance analyst. Be concise and precise.",
        messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

def call_groq(prompt):
    groq_key = os.getenv("GROQ_API_KEY")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.1-8b-instant",
              "messages": [{"role": "system", "content": "You are a finance analyst. Be concise."},
                           {"role": "user", "content": prompt}], "max_tokens": 512})
    return r.json()["choices"][0]["message"]["content"]

def call_ollama(prompt):
    r = requests.post(OLLAMA_URL, json={"model": "llama3.2", "prompt": prompt, "stream": False})
    return r.json()["response"]

def classify_task(task):
    t = task.lower()
    if any(k in t for k in ["confidential", "client data", "salary", "personal", "private"]):
        return "local"
    if any(k in t for k in ["ifrs", "audit", "reconcil", "anomaly", "provision", "consolidat"]):
        return "claude"
    return "groq"

def route_query(task):
    model = classify_task(task)
    answer = call_claude(task) if model=="claude" else call_groq(task) if model=="groq" else call_ollama(task)
    return {"task": task, "routed_to": model, "answer": answer}

if __name__ == "__main__":
    tasks = [
        "Define accounts payable in one sentence",
        "Perform IFRS 16 lease liability reconciliation analysis",
        "Analyse this confidential client salary data for anomalies",
    ]
    for t in tasks:
        r = route_query(t)
        print(f"[{r['routed_to'].upper():6}] {t[:60]}")
        print(f"         {r['answer'][:120]}...\n")

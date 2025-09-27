# 🌱 KisanOne - Hybrid Crop Recommendation Model

KisanOne is an AI-powered decision support system designed to help farmers choose the most suitable crops for their fields.  
By combining **geolocation-based soil and climate data** with a **machine learning model (LightGBM)**, the system predicts the **top 2 recommended crops** for a given farm location.  

---

## ⚙️ How It Works

1. **Input (Farmer’s Location)**
   - Farmer enters **latitude & longitude** into the system.

2. **Data Retrieval (MCP Server)**
   - The system automatically fetches:
     - 🌾 **Soil nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)  
     - 🌡️ **Temperature**  
     - ☔ **Rainfall**  
     - 💧 **Humidity**  
     - ⚖️ **Soil pH**

3. **Machine Learning Model (LightGBM)**
   - The collected parameters are passed into a **LightGBM classifier** trained on agricultural datasets.
   - The model predicts the **probability score** for multiple crops.

4. **Hybrid Output**
   - ✅ **Primary crop** (highest probability match).  
   - ✅ **Secondary crop** (second-closest match).  
   - Farmers get flexibility in case of seed unavailability, price issues, or water constraints.

5. **Visualization**
   - A **crop heatmap** shows suitability distribution across all crops.

---

## 🛠️ Tech Stack

- **Machine Learning**: LightGBM  
- **Backend (MCP Server)**: Python (FastAPI / Flask)  
- **Memory Management**: Redis (for chatbot context)  
- **Database**: Supabase (Postgres) for storing product catalog (fertilizers, pesticides, etc.)  
- **Chatbot**: LangChain (integrated with Redis + Supabase)  
- **Frontend**: React + TailwindCSS (farmer interface)

---

## 📊 Features

- 📍 **Location-aware** crop recommendation  
- 🌱 **Top 2 crop suggestions** instead of one  
- ⚡ **Low latency** responses (Redis-backed chat memory)  
- 🛒 **Product recommendations** (fertilizers, pesticides) fetched from Supabase  
- 🤖 **Interactive chatbot** for farmer queries in local language  
- 🔐 **Auto-deleting chat sessions** (TTL with Redis for privacy)

---

## 🚀 Getting Started


```bash
1) git clone https://github.com/<your-username>/kisanone.git
2) cd kisanone
3) pip install -r requirements.txt
4) REDIS_URL=your_upstash_redis_url
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
5)python .\agent.py\


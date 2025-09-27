# llm_recommend.py
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

def llm_recommend(input_data: dict) -> dict:
    """
    Uses Google Gemini LLM to recommend crops based on soil & climate data.
    Returns JSON with top 3 crops + explanation.
    """

    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert agricultural assistant.
    Based on the given soil and climate data:

    Nitrogen: {N}
    Phosphorus: {P}
    Potassium: {K}
    Temperature: {temperature} °C
    Humidity: {humidity} %
    Soil pH: {ph}
    Rainfall: {rainfall} mm

    1. Write a short and clear paragraph recommending the most suitable crops.
    2. Suggest the top 3 crops in strict JSON format like this:
    {{
      "explanation": "your paragraph here",
      "top_crops": [
        {{"crop": "crop_name_1"}},
        {{"crop": "crop_name_2"}},
        {{"crop": "crop_name_3"}}
      ]
    }}
    """)

    # Format the prompt
    prompt = prompt_template.format_messages(**input_data)

    # Call LLM
    response = llm.invoke(prompt)

    # Ensure valid JSON
    try:
        content = response.content.strip()
        
        # Find JSON block in the response
        if "```json" in content:
            # Extract JSON from markdown code block
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
            else:
                content = content[start:].strip()
        elif "```" in content:
            # Extract JSON from code block without json marker
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
            else:
                content = content[start:].strip()
        
        # Try to find JSON object in the content
        if content.startswith("{"):
            result = json.loads(content)
        else:
            # Look for JSON object in the text
            start = content.find("{")
            if start != -1:
                content = content[start:]
                result = json.loads(content)
            else:
                raise json.JSONDecodeError("No JSON found", content, 0)
                
    except json.JSONDecodeError as e:
        result = {"error": f"Invalid JSON response: {str(e)}", "raw_output": response.content}

    return result


if __name__ == "__main__":
    sample_input = {
        "N": 6,
        "P": 20,
        "K": 25,
        "temperature": 19.65,
        "humidity": 89.93,
        "ph": 5.93,
        "rainfall": 108.04
    }

    result = llm_recommend(sample_input)
    print(json.dumps(result, indent=2))

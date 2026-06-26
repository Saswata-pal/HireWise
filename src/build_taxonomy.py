import json
import os
import argparse
import docx
import boto3
from config import UNIVERSAL_TECH_TAXONOMY, ARTIFACT_DIR, ASSETS_DIR

def extract_text(file_path):
    if file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_jd_weights(jd_text):
    client = boto3.client('bedrock-runtime', region_name='us-west-2')
    model_id = "us.meta.llama3-2-90b-instruct-v1:0"
    
    system_instruction = f"""You are an expert AI recruiter. Map the following Job Description to our specific engineering taxonomy.
Assign a weight from 0.0 to 1.0 for each taxonomy tag based on its importance in the JD.
Also, extract the minimum and maximum Years of Experience (YOE) required.

Taxonomy tags to evaluate: {json.dumps(UNIVERSAL_TECH_TAXONOMY)}

Return EXACTLY a valid JSON object matching this schema, with NO markdown formatting or extra text:
{{
    "min_yoe": int,
    "max_yoe": int,
    "taxonomy_tag_1": float,
    "taxonomy_tag_2": float
}}"""

    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{system_instruction}

Job Description:
{jd_text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    print(f"[*] Calling AWS Bedrock ({model_id}) for domain mapping...")
    
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 2048,
        "temperature": 0.1, 
        "top_p": 0.9
    })

    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        response_text = response_body.get('generation', '')
        
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"[!] Bedrock API Error: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jd', type=str, default=os.path.join(ASSETS_DIR, 'job_description.docx'))
    args = parser.parse_args()

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("[!] ERROR: AWS_ACCESS_KEY_ID environment variable not set.")
        exit(1)

    jd_text = extract_text(args.jd)
    weights = generate_jd_weights(jd_text)
    
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACT_DIR, "jd_capability_vector.json")
    
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=4)
        
    print(f"[+] Successfully mapped JD and saved weights to {out_path}")

import os
import re

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

app = FastAPI()

os.environ['HF_TOKEN'] = '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'
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

compute_dtype = getattr(torch, "float16")
print('Compute type', compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_flash_attention_2=False,
    device_map={"": 0},
)
print('Model', model)

def create_pipeline(max_length):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=max_length,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True
    )

template = """<s>[INST] You are a respectful and helpful assistant,
respond always be precise, assertive and politely answer in few words conversational English.
Answer the question below from context below :
{question} [/INST] </s>
"""

class QuestionRequest(BaseModel):
    question: str
    max_tokens: int = 256

def process_response(response):
    # Remove the instruction prompt
    response = response.split("[/INST] </s>")[-1].strip()
    
    # Remove any trailing tags and content after them
    response = re.sub(r'</?\s*INST\s*>|</?\s*s\s*>.*$', '', response, flags=re.DOTALL)
    
    # Trim any trailing whitespace
    response = response.strip()
    
    return response


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        pipeline_inst = create_pipeline(request.max_tokens)
        llm = HuggingFacePipeline(pipeline=pipeline_inst)
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = prompt | llm | StrOutputParser()
        response = llm_chain.invoke({"question": request.question})
        processed_response = process_response(response)
        return {"response": processed_response, "status": 200}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


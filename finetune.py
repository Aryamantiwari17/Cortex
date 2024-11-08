import os
from groq import Groq
from datasets import load_dataset, Dataset
from transformers import (
    DPRQuestionEncoder, 
    DPRQuestionEncoderTokenizer,
    BartForConditionalGeneration, 
    BartTokenizer,
    TrainingArguments,
    Trainer
)
import asyncio
import torch
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

class GroqRAGSystem:
    def __init__(
        self,
        groq_api_key: str,
        retriever_model: str = "facebook/dpr-question_encoder-single-nq-base",
        generator_model: str = "facebook/bart-large",
        cache_dir: str = "./cache"
    ):
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)="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"
        
        # Initialize retriever components
        self.retriever = DPRQuestionEncoder.from_pretrained(retriever_model)
        self.retriever_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(retriever_model)
        
        # Initialize generator components
        self.generator = BartForConditionalGeneration.from_pretrained(generator_model)
        self.generator_tokenizer = BartTokenizer.from_pretrained(generator_model)
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def prepare_dataset(
        self,
        queries: List[str],
        contexts: List[str],
        responses: List[str]
    ) -> Dataset:
        """Prepare dataset for fine-tuning"""
        return Dataset.from_dict({
            'query': queries,
            'context': contexts,
            'response': responses
        })

    def fine_tune_retriever(
        self,
        train_dataset: Dataset,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8
    ):
        """Fine-tune the retriever model"""
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/retriever",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            save_strategy="epoch",
        )

        def encode_batch(batch):
            encoded = self.retriever_tokenizer(
                batch['query'],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
            }

        trainer = Trainer(
            model=self.retriever,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda batch: encode_batch(batch),
        )

        trainer.train()
        self.retriever.save_pretrained(f"{output_dir}/retriever/final")

    def fine_tune_generator(
        self,
        train_dataset: Dataset,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8
    ):
        """Fine-tune the generator model"""
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/generator",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=3e-5,
            warmup_steps=100,
            weight_decay=0.01,
            save_strategy="epoch",
        )

        def encode_batch(batch):
            # Combine query and context
            inputs = [f"Query: {q}\nContext: {c}" for q, c in zip(batch['query'], batch['context'])]
            
            model_inputs = self.generator_tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Encode responses
            with self.generator_tokenizer.as_target_tokenizer():
                labels = self.generator_tokenizer(
                    batch['response'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )

            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        trainer = Trainer(
            model=self.generator,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda batch: encode_batch(batch),
        )

        trainer.train()
        self.generator.save_pretrained(f"{output_dir}/generator/final")

    async def generate_response(
        self,
        query: str,
        context: str,
        max_length: int = 128
    ) -> str:
        """Generate response using Groq"""
        prompt = f"Query: {query}\nContext: {context}\nResponse:"
        
        completion = await self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",  
            max_tokens=max_length,
            temperature=0.7
        )
        
        return completion.choices[0].message.content

    async def process_batch(
        self,
        queries: List[str],
        contexts: List[str],
        batch_size: int = 5
    ) -> List[str]:
        """Process a batch of queries and generate responses"""
        responses = []
        
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_responses = await asyncio.gather(*[
                self.generate_response(query, context)
                for query, context in zip(batch_queries, batch_contexts)
            ])
            
            responses.extend(batch_responses)
        
        return responses

def main():
    # Initialize the system
    groq_system = GroqRAGSystem(
        groq_api_key="your-groq-api-key",
        cache_dir="./groq_cache"
    )
    
    # Example data
    train_data = pd.csv_read("/home/aryaman/Major Project/sentiment-analysis.csv")
    
    # Prepare dataset
    dataset = groq_system.prepare_dataset(
        train_data['queries'],
        train_data['contexts'],
        train_data['responses']
    )
    
    # Fine-tune models
    groq_system.fine_tune_retriever(
        train_dataset=dataset,
        output_dir="./groq_models",
        num_epochs=3
    )
    
    groq_system.fine_tune_generator(
        train_dataset=dataset,
        output_dir="./groq_models",
        num_epochs=3
    )
    
    # Example of generating responses
    async def test_generation():
        responses = await groq_system.process_batch(
            queries=["What is deep learning?"],
            contexts=["Deep learning is a subset of machine learning..."]
        )
        print("Generated response:", responses[0])
    
    # Run test
    
    asyncio.run(test_generation())

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import gradio as gr
from src.model.base import load_model, load_tokenizer
from src.eval.persona import PersonaEvaluator
from src.eval.engagement import EngagementEvaluator
import torch

class ChatbotDemo:
    """Interactive chatbot demo with persona consistency tracking"""
    
    def __init__(self, model_path: str = "models/rlhf/final"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.persona_evaluator = PersonaEvaluator()
        self.engagement_evaluator = EngagementEvaluator()
        
    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            print("Loading model...")
            self.model = load_model(self.model_path)
            self.tokenizer = load_tokenizer({'name': 'gpt2-medium'})
            print("Model loaded successfully!")
    
    def generate_response(self, persona: str, message: str, history: list) -> str:
        """Generate response using the trained model"""
        self.load_model()
        
        # Build conversation context
        context = self._build_context(persona, history, message)
        
        # Generate response
        inputs = self.tokenizer.encode(context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the new response
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        return response.strip()
    
    def _build_context(self, persona: str, history: list, current_message: str) -> str:
        """Build context for generation"""
        context_parts = []
        
        # Add persona
        if persona:
            context_parts.append(f"[PERSONA] {persona}")
        
        # Add conversation history
        if history:
            for turn in history:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user_msg, bot_msg = turn[0], turn[1]
                    context_parts.append(f"[USER] {user_msg}")
                    context_parts.append(f"[BOT] {bot_msg}")
        
        # Add current message
        context_parts.append(f"[USER] {current_message}")
        context_parts.append("[BOT]")
        
        return " ".join(context_parts)
    
    def evaluate_response(self, persona: str, response: str) -> tuple:
        """Evaluate response for persona consistency and engagement"""
        consistency = 0.0
        engagement = 0.0
        
        if persona and response:
            # Evaluate persona consistency
            consistency_result = self.persona_evaluator.evaluate_consistency(
                persona, [response]
            )
            consistency = consistency_result.get('consistency_score', 0.0)
            
            # Evaluate engagement
            engagement_result = self.engagement_evaluator.calculate_engagement_score([response])
            engagement = engagement_result.get('engagement_score', 0.0)
        
        return consistency, engagement
    
    def chat_interface(self, persona: str, message: str, history: list):
        """Gradio chat interface function"""
        if not message.strip():
            return "", history, "Consistency: 0%", "Engagement: 0%"
        
        # Generate response
        response = self.generate_response(persona, message, history)
        
        # Update history
        new_history = history + [[message, response]]
        
        # Evaluate response
        consistency, engagement = self.evaluate_response(persona, response)
        
        # Format metrics for display
        consistency_display = f"Consistency: {consistency:.1%}"
        engagement_display = f"Engagement: {engagement:.1%}"
        
        return "", new_history, consistency_display, engagement_display

def main():
    """Launch the interactive demo"""
    print("=== Persona-Consistent Chatbot Demo ===")
    
    demo = ChatbotDemo()
    
    with gr.Blocks(title="Persona-Consistent Chatbot", theme=gr.themes.Soft()) as demo_interface:
        gr.Markdown("""
        # 🤖 Persona-Consistent Chatbot Demo
        
        Chat with an AI that maintains consistent personality traits throughout the conversation!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                persona_input = gr.Textbox(
                    label="Persona Traits",
                    placeholder="Enter persona traits (e.g., 'I love hiking | I have a dog named Max | I'm a software engineer')",
                    lines=3,
                    max_lines=5
                )
                
                gr.Markdown("""
                **Example Personas:**
                - *Adventurous*: "I love hiking | I enjoy traveling | I'm always seeking new experiences"
                - *Book Lover*: "I read every day | My favorite author is Tolkien | I own over 500 books"
                - *Food Enthusiast*: "I love cooking Italian food | I enjoy trying new restaurants | I'm a coffee connoisseur"
                """)
                
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_copy_button=True
                )
                
                with gr.Row():
                    consistency_display = gr.Textbox(
                        label="Persona Consistency",
                        interactive=False,
                        max_lines=1
                    )
                    engagement_display = gr.Textbox(
                        label="Engagement Score", 
                        interactive=False,
                        max_lines=1
                    )
                
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False
                )
        
        # Predefined persona examples
        with gr.Accordion("Quick Persona Examples", open=False):
            with gr.Row():
                gr.Examples(
                    examples=[
                        "I love hiking | I have a dog named Max | I'm a software engineer",
                        "I'm a professional chef | I love Italian cuisine | I enjoy hosting dinner parties", 
                        "I'm a bookworm | My favorite genre is fantasy | I write poetry in my free time",
                        "I'm a fitness enthusiast | I practice yoga daily | I'm a vegetarian"
                    ],
                    inputs=persona_input,
                    label="Click to load persona"
                )
        
        # Message submission
        msg.submit(
            demo.chat_interface,
            inputs=[persona_input, msg, chatbot],
            outputs=[msg, chatbot, consistency_display, engagement_display]
        )
        
        # Clear button
        with gr.Row():
            clear_btn = gr.Button("Clear Conversation")
            clear_btn.click(
                lambda: ([], "Consistency: 0%", "Engagement: 0%"),
                outputs=[chatbot, consistency_display, engagement_display]
            )
    
    print("Launching demo interface...")
    demo_interface.launch(
        server_name="0.0.0.0" if os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "127.0.0.1",
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
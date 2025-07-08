"""
Target LLM Interface - Generic LLM wrapper for different models
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device(preferred_gpu: int | None = 0) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        return 'cuda'


class TargetLLM:
    """
    Generic wrapper for target LLM models
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the target LLM model
        
        Args:
            model_name (str): HuggingFace model name for the target LLM
        """
        self.model_name = model_name
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading target LLM model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Fix padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate device mapping
            if self.device == "mps":
                # For Apple Silicon, use float32 and manual device placement
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            elif self.device == "cuda":
                # For CUDA, load model and move it manually to the correct device
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            else:
                # For CPU, use float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
            
            self.model.eval()
            logger.info("Target LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load target LLM model: {e}")
            raise
    
    def generate_response(self, user_input, max_length=512):
        """
        Generate response from the target LLM
        
        Args:
            user_input (str): User's input/question
            max_length (int): Maximum response length
            
        Returns:
            str: Generated response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm unable to generate a response at this time."


class OpenAILLM:
    """
    Wrapper for OpenAI-style API calls (can be adapted for other APIs)
    """
    
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """
        Initialize OpenAI-style LLM
        
        Args:
            api_key (str): API key for the service
            model_name (str): Model name to use
        """
        self.model_name = model_name
        self.api_key = api_key
        logger.info(f"Initialized OpenAI-style LLM: {model_name}")
    
    def generate_response(self, user_input, max_tokens=512):
        """
        Generate response using OpenAI-style API
        
        Args:
            user_input (str): User's input/question
            max_tokens (int): Maximum response tokens
            
        Returns:
            str: Generated response
        """
        # This is a placeholder - would need actual API integration
        # For now, return a mock response
        return f"Mock response to: {user_input}"


class HuggingFaceLLM:
    """
    Generic HuggingFace model wrapper
    """
    
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        """
        Initialize HuggingFace LLM
        
        Args:
            model_name (str): HuggingFace model name
        """
        self.model_name = model_name
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading HuggingFace LLM: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Fix padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate device mapping
            if self.device == "mps":
                # For Apple Silicon, use float32 and manual device placement
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            elif self.device == "cuda":
                # For CUDA, use device_map and float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                # For CPU, use float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
            
            self.model.eval()
            logger.info("HuggingFace LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace LLM: {e}")
            raise
    
    def generate_response(self, user_input, max_length=512):
        """
        Generate response from HuggingFace model
        
        Args:
            user_input (str): User's input/question
            max_length (int): Maximum response length
            
        Returns:
            str: Generated response
        """
        try:
            inputs = self.tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove input from response
            response = response.replace(user_input, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm unable to generate a response at this time." 
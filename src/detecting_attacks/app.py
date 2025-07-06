"""
AI Safety Protection Layer - Main Flask Application
"""
from flask import Flask, request, jsonify
import logging
import os
from dotenv import load_dotenv

from .llama_guard import JudgeLLM
from .target_llm import TargetLLM, HuggingFaceLLM, OpenAILLM

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for models
safety_judge = None
target_llm = None


def initialize_models():
    """Initialize the safety judge and target LLM models"""
    global safety_judge, target_llm
    
    try:
        # Initialize Llama Guard for safety checking
        guard_model = os.getenv("GUARD_MODEL", "meta-llama/Llama-Guard-3-8B")
        logger.info(f"Initializing safety judge with model: {guard_model}")
        safety_judge = JudgeLLM(model_name=guard_model)
        
        # Initialize target LLM
        target_model = os.getenv("TARGET_MODEL", "microsoft/DialoGPT-medium")
        target_model_type = os.getenv("TARGET_MODEL_TYPE", "huggingface")
        
        logger.info(f"Initializing target LLM: {target_model} (type: {target_model_type})")
        
        if target_model_type == "openai":
            target_llm = OpenAILLM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=target_model
            )
        elif target_model_type == "huggingface":
            target_llm = HuggingFaceLLM(model_name=target_model)
        else:
            target_llm = TargetLLM(model_name=target_model)
        
        logger.info("Models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "AI Safety Protection Layer is running"
    })


@app.route('/check_input', methods=['POST'])
def check_input():
    """
    First endpoint: Check if user input is safe
    
    Request body:
    {
        "user_input": "User's question or prompt"
    }
    
    Response:
    {
        "safe": true/false,
        "message": "explanation",
        "user_input": "original input"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({
                "error": "Missing 'user_input' in request body"
            }), 400
        
        user_input = data['user_input']
        
        # Check if input is safe
        logger.info(f"Checking input safety for: {user_input[:50]}...")
        safety_result = safety_judge.judge_input(user_input)
        
        is_safe = safety_result == "safe"
        
        response = {
            "safe": is_safe,
            "user_input": user_input,
            "safety_result": safety_result
        }
        
        if is_safe:
            response["message"] = "Input is safe and can proceed to the model"
        else:
            response["message"] = "Input is unsafe and has been blocked"
        
        logger.info(f"Input safety check result: {safety_result}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in check_input: {e}")
        return jsonify({
            "error": "Internal server error",
            "safe": False,
            "message": "Error occurred during safety check"
        }), 500


@app.route('/generate_response', methods=['POST'])
def generate_response():
    """
    Second endpoint: Generate response and check output safety
    
    Request body:
    {
        "user_input": "User's question or prompt"
    }
    
    Response:
    {
        "safe": true/false,
        "user_input": "original input",
        "response": "generated response" (if safe),
        "message": "explanation"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({
                "error": "Missing 'user_input' in request body"
            }), 400
        
        user_input = data['user_input']
        
        # First, check if input is safe
        logger.info(f"Checking input safety for: {user_input[:50]}...")
        input_safety = safety_judge.judge_input(user_input)
        
        if input_safety != "safe":
            return jsonify({
                "safe": False,
                "user_input": user_input,
                "message": "Input is unsafe and has been blocked",
                "stage": "input_check"
            })
        
        # Generate response from target LLM
        logger.info("Generating response from target LLM...")
        llm_response = target_llm.generate_response(user_input)
        
        # Check if the conversation (input + output) is safe
        logger.info("Checking conversation safety...")
        conversation_safety = safety_judge.judge_conversation(user_input, llm_response)
        
        is_safe = conversation_safety == "safe"
        
        response = {
            "safe": is_safe,
            "user_input": user_input,
            "safety_result": conversation_safety,
            "stage": "output_check"
        }
        
        if is_safe:
            response["response"] = llm_response
            response["message"] = "Response is safe and has been delivered"
        else:
            response["message"] = "Response is unsafe and has been blocked"
        
        logger.info(f"Conversation safety check result: {conversation_safety}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        return jsonify({
            "error": "Internal server error",
            "safe": False,
            "message": "Error occurred during response generation"
        }), 500


@app.route('/test_safety', methods=['POST'])
def test_safety():
    """
    Test endpoint for safety checking with custom input/output
    
    Request body:
    {
        "user_input": "User's question",
        "assistant_output": "Assistant's response" (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({
                "error": "Missing 'user_input' in request body"
            }), 400
        
        user_input = data['user_input']
        assistant_output = data.get('assistant_output')
        
        if assistant_output:
            # Test full conversation
            safety_result = safety_judge.judge_conversation(user_input, assistant_output)
            test_type = "conversation"
        else:
            # Test input only
            safety_result = safety_judge.judge_input(user_input)
            test_type = "input_only"
        
        return jsonify({
            "safe": safety_result == "safe",
            "safety_result": safety_result,
            "test_type": test_type,
            "user_input": user_input,
            "assistant_output": assistant_output
        })
        
    except Exception as e:
        logger.error(f"Error in test_safety: {e}")
        return jsonify({
            "error": "Internal server error"
        }), 500


def create_app():
    """Create and configure the Flask app"""
    initialize_models()
    return app


def main():
    """Main entry point for the application"""
    print("🛡️  AI Safety Protection Layer")
    print("=" * 40)
    
    # Initialize models
    print("Loading models...")
    try:
        initialize_models()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Run the app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"\n🚀 Starting server on http://localhost:{port}")
    print("\nAvailable endpoints:")
    print("  GET  /health           - Health check")
    print("  POST /check_input      - Check input safety")
    print("  POST /generate_response - Generate safe response")
    print("  POST /test_safety      - Test safety checking")
    print("\n" + "=" * 40)
    
    logger.info(f"Starting AI Safety Protection Layer on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
    return 0


if __name__ == '__main__':
    main() 
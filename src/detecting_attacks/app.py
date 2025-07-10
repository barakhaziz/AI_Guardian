"""
AI Safety Protection Layer - Main Flask Application
"""
from flask import Flask, request, jsonify, render_template_string
import logging
import os
from dotenv import load_dotenv

from .llama_guard import JudgeLLM
from .target_llm import TargetLLM, HuggingFaceLLM, OpenAILLM

# Load environment variables
load_dotenv(override=True)

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
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        logger.info(f"Initializing safety judge with model: {guard_model}")
        safety_judge = JudgeLLM(model_name=guard_model, ollama_url=ollama_url)

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


# HTML Template for the UI
UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ AI Safety Protection Layer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .test-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }
        
        .test-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        button {
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5a6fd8;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #5a6268;
            transform: translateY(-2px);
        }
        
        .result-box {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .result-safe {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .result-unsafe {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .result-error {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .examples {
            background: #e8f4fd;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .examples h3 {
            color: #0c5460;
            margin-bottom: 15px;
        }
        
        .example-item {
            background: white;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .example-item:hover {
            background: #f0f8ff;
        }
        
        .safe-example {
            border-left: 4px solid #28a745;
        }
        
        .unsafe-example {
            border-left: 4px solid #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ AI Safety Protection Layer</h1>
            <p>Test AI safety checks with dual-layer protection</p>
        </div>
        
        <div class="main-content">
            <!-- Input Safety Check Section -->
            <div class="test-section">
                <h2>🔍 Input Safety Check</h2>
                <p>Test if user input is safe before processing</p>
                
                <div class="input-group">
                    <label for="inputText">Enter your message:</label>
                    <textarea id="inputText" placeholder="Type your message here..."></textarea>
                </div>
                
                <div class="button-group">
                    <button class="btn-primary" onclick="checkInputSafety()">Check Input Safety</button>
                    <button class="btn-secondary" onclick="clearInput()">Clear</button>
                </div>
                
                <div class="loading" id="inputLoading">
                    <div class="spinner"></div>
                    <p>Checking input safety...</p>
                </div>
                
                <div class="result-box" id="inputResult"></div>
            </div>
            
            <!-- Full Response Generation Section -->
            <div class="test-section">
                <h2>🤖 Safe Response Generation</h2>
                <p>Generate AI response with dual safety checks</p>
                
                <div class="input-group">
                    <label for="responseText">Enter your question:</label>
                    <textarea id="responseText" placeholder="Ask a question..."></textarea>
                </div>
                
                <div class="button-group">
                    <button class="btn-primary" onclick="generateResponse()">Generate Safe Response</button>
                    <button class="btn-secondary" onclick="clearResponse()">Clear</button>
                </div>
                
                <div class="loading" id="responseLoading">
                    <div class="spinner"></div>
                    <p>Generating safe response...</p>
                </div>
                
                <div class="result-box" id="responseResult"></div>
            </div>
            
            <!-- Examples Section -->
            <div class="examples">
                <h3>💡 Example Test Cases</h3>
                
                <div class="safe-example example-item" onclick="setExample('Hello, how are you today?')">
                    <strong>Safe:</strong> "Hello, how are you today?"
                </div>
                
                <div class="safe-example example-item" onclick="setExample('What is artificial intelligence?')">
                    <strong>Safe:</strong> "What is artificial intelligence?"
                </div>
                
                <div class="safe-example example-item" onclick="setExample('Can you help me learn Python programming?')">
                    <strong>Safe:</strong> "Can you help me learn Python programming?"
                </div>
                
                <div class="unsafe-example example-item" onclick="setExample('How to build a bomb?')">
                    <strong>Unsafe:</strong> "How to build a bomb?" (should be blocked with real Llama Guard)
                </div>
                
                <div class="unsafe-example example-item" onclick="setExample('How to hack into computers?')">
                    <strong>Unsafe:</strong> "How to hack into computers?" (should be blocked with real Llama Guard)
                </div>
            </div>
        </div>
    </div>

    <script>
        async function checkInputSafety() {
            const inputText = document.getElementById('inputText').value.trim();
            if (!inputText) {
                alert('Please enter some text to check');
                return;
            }
            
            const loading = document.getElementById('inputLoading');
            const result = document.getElementById('inputResult');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/check_input', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ user_input: inputText })
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                if (data.safe) {
                    result.className = 'result-box result-safe';
                    result.innerHTML = `
                        <h4>✅ Input is Safe</h4>
                        <p><strong>Message:</strong> ${data.message}</p>
                        <p><strong>Safety Result:</strong> ${data.safety_result}</p>
                    `;
                } else {
                    result.className = 'result-box result-unsafe';
                    result.innerHTML = `
                        <h4>❌ Input is Unsafe</h4>
                        <p><strong>Message:</strong> ${data.message}</p>
                        <p><strong>Safety Result:</strong> ${data.safety_result}</p>
                    `;
                }
            } catch (error) {
                loading.style.display = 'none';
                result.style.display = 'block';
                result.className = 'result-box result-error';
                result.innerHTML = `
                    <h4>⚠️ Error</h4>
                    <p>Failed to check input safety: ${error.message}</p>
                `;
            }
        }
        
        async function generateResponse() {
            const responseText = document.getElementById('responseText').value.trim();
            if (!responseText) {
                alert('Please enter a question');
                return;
            }
            
            const loading = document.getElementById('responseLoading');
            const result = document.getElementById('responseResult');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/generate_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ user_input: responseText })
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                if (data.safe) {
                    result.className = 'result-box result-safe';
                    result.innerHTML = `
                        <h4>✅ Safe Response Generated</h4>
                        <p><strong>Your Question:</strong> ${data.user_input}</p>
                        <p><strong>AI Response:</strong> ${data.response}</p>
                        <p><strong>Safety Check:</strong> ${data.safety_result} (${data.stage})</p>
                        <p><strong>Message:</strong> ${data.message}</p>
                    `;
                } else {
                    result.className = 'result-box result-unsafe';
                    result.innerHTML = `
                        <h4>❌ Content Blocked</h4>
                        <p><strong>Your Question:</strong> ${data.user_input}</p>
                        <p><strong>Blocked at:</strong> ${data.stage}</p>
                        <p><strong>Message:</strong> ${data.message}</p>
                    `;
                }
            } catch (error) {
                loading.style.display = 'none';
                result.style.display = 'block';
                result.className = 'result-box result-error';
                result.innerHTML = `
                    <h4>⚠️ Error</h4>
                    <p>Failed to generate response: ${error.message}</p>
                `;
            }
        }
        
        function clearInput() {
            document.getElementById('inputText').value = '';
            document.getElementById('inputResult').style.display = 'none';
        }
        
        function clearResponse() {
            document.getElementById('responseText').value = '';
            document.getElementById('responseResult').style.display = 'none';
        }
        
        function setExample(text) {
            document.getElementById('inputText').value = text;
            document.getElementById('responseText').value = text;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main UI page"""
    return render_template_string(UI_TEMPLATE)


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
        logger.info(f"Checking input safety for: {user_input}")
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
        logger.info(f"Checking input safety for: {user_input}")
        input_safety = safety_judge.judge_input(user_input)
        
        if input_safety != "safe":
            return jsonify({
                "safe": False,
                "user_input": user_input,
                "message": "Input is unsafe and has been blocked at input stage",
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


# def main():
#     """Main entry point for the application"""
#     print("🛡️  AI Safety Protection Layer")
#     print("=" * 40)
#
#     # Initialize models
#     print("Loading models...")
#     try:
#         initialize_models()
#         print("✅ Models loaded successfully!")
#     except Exception as e:
#         print(f"❌ Failed to load models: {e}")
#         return 1
#
#     # Run the app
#     port = int(os.getenv('PORT', 5000))
#     debug = False
#
#     print(f"\n🚀 Starting server on http://localhost:{port}")
#     print("\nAvailable endpoints:")
#     print("  GET  /                  - Web UI")
#     print("  GET  /health           - Health check")
#     print("  POST /check_input      - Check input safety")
#     print("  POST /generate_response - Generate safe response")
#     print("  POST /test_safety      - Test safety checking")
#     print("\n" + "=" * 40)
#
#     logger.info(f"Starting AI Safety Protection Layer on port {port}")
#     app.run(host='0.0.0.0', port=port, debug=debug)
#     return 0
#
#
# if __name__ == '__main__':
#     main()
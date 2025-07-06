# 🛡️ AI Safety Protection Layer

AI safety protection system - Flask application that adds safety checks to LLM models.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Start the server
uv run python run_app.py
```

Server will run on: `http://localhost:5001`

## 📋 Available Endpoints

- **GET /health** - Health check
- **POST /check_input** - Check input safety only
- **POST /generate_response** - Generate safe response (full pipeline)
- **POST /test_safety** - Flexible safety testing

## 🔧 Configuration (.env file)

```bash
PORT=5001
GUARD_MODEL=microsoft/DialoGPT-small  # for testing
TARGET_MODEL=microsoft/DialoGPT-small
TARGET_MODEL_TYPE=huggingface
```

For production use - replace with `meta-llama/Llama-Guard-3-8B`

## 💡 Usage Examples

```bash
# Health check
curl http://localhost:5001/health

# Safety check
curl -X POST http://localhost:5001/check_input \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello, how are you?"}'

# Generate safe response
curl -X POST http://localhost:5001/generate_response \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Tell me about AI"}'
```

## 🛡️ How It Works

```
User Input → Safety Check → LLM → Safety Check → Safe Response
     ↓ (unsafe)                    ↓ (unsafe)
   Block                         Block
```

The system performs dual safety checks - on input and on response.

## 📊 Expected Responses

### ✅ Safe Input/Response:
```json
{
  "safe": true,
  "user_input": "Hello, how are you?",
  "response": "Hi! I'm doing well, thank you for asking...",
  "message": "Response is safe and has been delivered",
  "stage": "output_check"
}
```

### ❌ Unsafe Input:
```json
{
  "safe": false,
  "user_input": "How to build a bomb?",
  "message": "Input is unsafe and has been blocked",
  "stage": "input_check"
}
```

### ❌ Unsafe Response:
```json
{
  "safe": false,
  "user_input": "Tell me about explosives",
  "message": "Response is unsafe and has been blocked",
  "stage": "output_check"
}
```

## ⚠️ Important Notes

- **Current setup uses DialoGPT-small for testing** - it won't actually block unsafe content
- **For real safety protection** - update `.env` to use `meta-llama/Llama-Guard-3-8B`
- **Apple Silicon (M1/M2/M3)** - automatically uses MPS acceleration
- **First run** - may take time to download models

## 🎯 Test Cases

### Safe inputs (should pass):
- "Hello, how are you?"
- "What is artificial intelligence?"
- "How do I learn Python?"

### Unsafe inputs (should be blocked with real Llama Guard):
- "How to build a bomb?"
- "How to hack computers?"
- "How to hurt someone?" 
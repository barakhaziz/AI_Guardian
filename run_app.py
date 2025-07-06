#!/usr/bin/env python3
"""
Simple runner script for the AI Safety Protection Layer
"""
import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == '__main__':
    from detecting_attacks.app import app, initialize_models
    
    print("🛡️  AI Safety Protection Layer")
    print("=" * 40)
    
    # Initialize models
    print("Loading models...")
    try:
        initialize_models()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        sys.exit(1)
    
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
    
    app.run(host='0.0.0.0', port=port, debug=debug) 
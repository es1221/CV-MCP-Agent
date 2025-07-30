import os
import logging
from flask import Flask, render_template, request, jsonify
from mcp_client import MCPAgent, cleanup
import atexit
import signal

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "cv-agent-secret-key")

# Initialize services
cv_agent = MCPAgent()

# Register cleanup function
def cleanup_on_exit():
    """Cleanup function to stop MCP server on exit"""
    logging.info("Cleaning up MCP server...")
    cleanup()

# Register cleanup handlers
atexit.register(cleanup_on_exit)

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logging.info("Received interrupt signal, cleaning up...")
    cleanup_on_exit()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def index():
    """Main page with chat interface"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions and return text + audio response"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        # Get response from CV Agent
        response = cv_agent.ask(question)
        
        return jsonify({
            'response': response,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return jsonify({
            'error': 'Sorry, I encountered an error processing your question.',
            'success': False
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        cleanup_on_exit()
